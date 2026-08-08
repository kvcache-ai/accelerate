# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.utils.dataclasses import FP8BackendType
from accelerate.utils.fsdp_utils import fsdp2_load_full_state_dict


class ModelWithPersistentBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("router_bias", torch.tensor([3.0, 4.0]))


class AdapterStateModel(torch.nn.Module):
    def __init__(self, omit_placeholder=False):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([1.0, 2.0]), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        self.placeholder = torch.nn.Parameter(torch.tensor([5.0, 6.0]), requires_grad=False)
        self.omit_placeholder = omit_placeholder

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.omit_placeholder:
            state_dict.pop("placeholder", None)
        return state_dict


def _fsdp2_accelerator_stub():
    return SimpleNamespace(distributed_type=DistributedType.FSDP, is_fsdp2=True)


class StagedPrepareHarness:
    device_placement = False
    distributed_type = DistributedType.FSDP
    is_fsdp2 = True
    parallelism_config = None
    fp8_backend = FP8BackendType.NO

    def __init__(self):
        self.state = SimpleNamespace(kt_config=None)
        self._models = []
        self._optimizers = []
        self._dataloaders = []
        self._schedulers = []

    def verify_device_map(self, model):
        return False

    def _validate_fsdp2_prepare_inputs(self, models, optimizers):
        return Accelerator._validate_fsdp2_prepare_inputs(self, models, optimizers)

    def _prepare_fsdp2(self, *objects):
        for item in objects:
            if isinstance(item, torch.nn.Module):
                self._models.append(item)
            elif isinstance(item, torch.optim.Optimizer):
                self._optimizers.append(item)
        return list(objects)


def _run_two_rank_state_dict_checks(rank, world_size, rendezvous_path):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        buffer_model = ModelWithPersistentBuffer()
        full_state_dict = buffer_model.state_dict() if rank == 0 else {}
        buffer_model.to(torch.device("meta"))
        load_accelerator = SimpleNamespace(is_main_process=rank == 0, device=torch.device("cpu"))

        fsdp2_load_full_state_dict(load_accelerator, buffer_model, full_state_dict)

        torch.testing.assert_close(buffer_model.router_bias, torch.tensor([3.0, 4.0]))
        assert not buffer_model.router_bias.is_meta

        from torch.distributed.fsdp import fully_shard

        adapter_model = AdapterStateModel()
        fully_shard(adapter_model, ignored_params={adapter_model.placeholder})
        adapter_model.register_parameter("late_adapter", torch.nn.Parameter(torch.tensor([7.0, 8.0])))
        rank_local_parameter = torch.nn.Parameter(torch.tensor([9.0, 10.0]))
        staged_optimizer = torch.optim.AdamW(
            [
                {"params": adapter_model.parameters()},
                {"params": [rank_local_parameter]},
            ]
        )
        Accelerator._validate_fsdp2_prepare_inputs(
            SimpleNamespace(_models=[adapter_model]),
            [],
            [staged_optimizer],
        )
        state_dict = Accelerator.get_state_dict(
            _fsdp2_accelerator_stub(),
            adapter_model,
            adapter_only=True,
            excluded_parameter_names=("placeholder",),
        )

        if rank == 0:
            assert set(state_dict) == {"adapter", "late_adapter"}
            torch.testing.assert_close(state_dict["adapter"], torch.tensor([3.0, 4.0]))
            torch.testing.assert_close(state_dict["late_adapter"], torch.tensor([7.0, 8.0]))
        else:
            assert state_dict == {}
        assert not adapter_model.placeholder.requires_grad

        excluded_names = ("placeholder",) if rank == 0 else ("missing",)
        with pytest.raises(RuntimeError, match="rank 1: ValueError"):
            Accelerator.get_state_dict(
                _fsdp2_accelerator_stub(),
                adapter_model,
                adapter_only=True,
                excluded_parameter_names=excluded_names,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("is_main_process", [True, False])
def test_fsdp2_load_full_state_dict_materializes_persistent_buffers(is_main_process):
    model = ModelWithPersistentBuffer()
    full_state_dict = model.state_dict() if is_main_process else {}
    model.to(torch.device("meta"))
    accelerator = SimpleNamespace(is_main_process=is_main_process, device=torch.device("cpu"))

    def broadcast(tensor, src, group):
        if not is_main_process:
            tensor.copy_(torch.tensor([3.0, 4.0]))

    with patch("torch.distributed.broadcast", side_effect=broadcast) as mock_broadcast:
        fsdp2_load_full_state_dict(accelerator, model, full_state_dict)

    mock_broadcast.assert_called_once()
    torch.testing.assert_close(model.router_bias, torch.tensor([3.0, 4.0]))
    assert model.router_bias.device.type == "cpu"
    assert model.weight.device.type == ("cpu" if is_main_process else "meta")


def test_fsdp2_adapter_only_state_dict_handles_omitted_exclusion():
    model = AdapterStateModel(omit_placeholder=True)

    state_dict = Accelerator.get_state_dict(
        _fsdp2_accelerator_stub(),
        model,
        adapter_only=True,
        excluded_parameter_names=("placeholder",),
    )

    assert set(state_dict) == {"adapter"}
    torch.testing.assert_close(state_dict["adapter"], torch.tensor([3.0, 4.0]))
    assert not model.placeholder.requires_grad


@pytest.mark.parametrize(
    ("adapter_only", "excluded_parameter_names"),
    [("yes", ()), (True, None), (True, "placeholder")],
)
def test_fsdp2_state_dict_validation_uses_synchronized_preflight(adapter_only, excluded_parameter_names):
    with pytest.raises(RuntimeError, match="FSDP2 state-dict preflight failed"):
        Accelerator.get_state_dict(
            _fsdp2_accelerator_stub(),
            AdapterStateModel(),
            adapter_only=adapter_only,
            excluded_parameter_names=excluded_parameter_names,
        )


def test_fsdp2_staged_optimizer_must_reference_prepared_model():
    model = AdapterStateModel()
    accelerator = SimpleNamespace(_models=[model])
    optimizer = torch.optim.AdamW(model.parameters())

    Accelerator._validate_fsdp2_prepare_inputs(accelerator, [], [optimizer])

    rank_local_parameter = torch.nn.Parameter(torch.ones(1))
    mixed_optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": [rank_local_parameter]},
        ]
    )
    Accelerator._validate_fsdp2_prepare_inputs(accelerator, [], [mixed_optimizer])

    unrelated_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    with pytest.raises(ValueError, match="does not reference any parameter from the prepared model"):
        Accelerator._validate_fsdp2_prepare_inputs(accelerator, [], [unrelated_optimizer])


def test_fsdp2_staged_optimizer_requires_prepared_model():
    model = AdapterStateModel()
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(ValueError, match="requires exactly one model"):
        Accelerator._validate_fsdp2_prepare_inputs(SimpleNamespace(_models=[]), [], [optimizer])


def test_fsdp2_prepare_accepts_model_then_optimizer():
    accelerator = StagedPrepareHarness()
    model = AdapterStateModel()

    prepared_model = Accelerator.prepare(accelerator, model)
    prepared_model.register_parameter("late_adapter", torch.nn.Parameter(torch.ones(1)))
    optimizer = torch.optim.AdamW(prepared_model.parameters())
    prepared_optimizer = Accelerator.prepare(accelerator, optimizer)

    assert prepared_model is model
    assert prepared_optimizer is optimizer
    assert accelerator._models == [model]
    assert accelerator._optimizers == [optimizer]


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires torch.distributed gloo")
def test_fsdp2_state_dict_contract_is_symmetric_across_two_ranks():
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous_path = f"{temporary_directory}/rendezvous"
        mp.spawn(_run_two_rank_state_dict_checks, args=(2, rendezvous_path), nprocs=2, join=True)

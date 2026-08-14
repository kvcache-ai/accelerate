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

import gc
import os
import tempfile
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from accelerate import Accelerator
from accelerate.utils import DistributedType, DynamoBackend
from accelerate.utils.dataclasses import FP8BackendType
from accelerate.utils.fsdp_utils import (
    fsdp2_load_full_state_dict,
    fsdp2_prepare_model,
    fsdp2_switch_optimizer_parameters,
)


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
        placeholder = torch.tensor([5.0]).as_strided((2,), (0,)) if omit_placeholder else torch.tensor([5.0, 6.0])
        self.placeholder = torch.nn.Parameter(placeholder, requires_grad=False)
        self.omit_placeholder = omit_placeholder

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.omit_placeholder:
            state_dict.pop("placeholder", None)
        return state_dict


class GradientClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


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

    def _validate_kt_distributed_setup(self, objects):
        pass

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

        torch.testing.assert_close(buffer_model.weight, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(buffer_model.router_bias, torch.tensor([3.0, 4.0]))
        assert not buffer_model.weight.is_meta
        assert not buffer_model.router_bias.is_meta

        from torch.distributed.fsdp import fully_shard

        ignored_model = AdapterStateModel()
        ignored_full_state_dict = ignored_model.state_dict() if rank == 0 else {}
        ignored_model.to(torch.device("meta"))
        fully_shard(ignored_model, ignored_params={ignored_model.placeholder})
        fsdp2_load_full_state_dict(load_accelerator, ignored_model, ignored_full_state_dict)
        torch.testing.assert_close(ignored_model.placeholder, torch.tensor([5.0, 6.0]))
        assert not ignored_model.placeholder.is_meta

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

        rank_local_names = ("placeholder",) if rank == 0 else ()
        with pytest.raises(RuntimeError, match="must be identical on every rank"):
            fsdp2_load_full_state_dict(
                load_accelerator,
                AdapterStateModel(),
                AdapterStateModel().state_dict() if rank == 0 else {},
                rank_local_parameter_names=rank_local_names,
            )
    finally:
        dist.destroy_process_group()


def _run_two_rank_gradient_clip_checks(rank, world_size, rendezvous_path):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torch.distributed.fsdp import fully_shard

        model = GradientClipModel()
        fully_shard(model)
        model.weight.grad = torch.zeros_like(model.weight)
        model.weight.grad.to_local().fill_(3.0 if rank == 0 else 0.0)

        external_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        external_parameter.grad = torch.tensor([4.0], dtype=torch.bfloat16)
        rank_local_parameters = (external_parameter,) if rank == 0 else ()
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            distributed_type=DistributedType.FSDP,
            is_fsdp2=True,
            _models=[model],
            unscale_gradients=lambda: None,
        )

        total_norm = Accelerator.clip_grad_norm_(
            accelerator,
            model.parameters(),
            max_norm=1.0,
            rank_local_parameters=rank_local_parameters,
        )

        torch.testing.assert_close(total_norm, total_norm.new_tensor(5.0))
        expected_coefficient = 1.0 / (5.0 + 1e-6)
        model_local_grad = model.weight.grad.to_local()
        torch.testing.assert_close(
            model_local_grad,
            model_local_grad.new_tensor([3.0 * expected_coefficient if rank == 0 else 0.0]),
        )
        if rank == 0:
            torch.testing.assert_close(
                external_parameter.grad.float(),
                torch.tensor([4.0 * expected_coefficient]),
                atol=0.005,
                rtol=0,
            )

        invalid_rank_local_parameters = (external_parameter, external_parameter) if rank == 0 else ()
        with pytest.raises(RuntimeError, match="rank 0: ValueError.*must not contain duplicates"):
            Accelerator.clip_grad_norm_(
                accelerator,
                model.parameters(),
                max_norm=1.0,
                rank_local_parameters=invalid_rank_local_parameters,
            )
    finally:
        dist.destroy_process_group()


def _run_two_rank_adapter_checkpoint_checks(rank, world_size, rendezvous_path, output_dir, omit_placeholder):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        from accelerate.utils.fsdp_utils import load_fsdp_model, save_fsdp_model

        model = AdapterStateModel(omit_placeholder=omit_placeholder)
        if omit_placeholder:
            fully_shard(model, ignored_params={model.placeholder})
        else:
            fully_shard(model)
        with torch.no_grad():
            model.adapter.to_local().fill_(rank + 3)
        expected_adapter = model.adapter.full_tensor().detach().clone()

        plugin = SimpleNamespace(
            fsdp_version=2,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=SimpleNamespace(offload_to_cpu=False, rank0_only=False),
            optim_state_dict_config=None,
        )
        accelerator = SimpleNamespace(
            num_processes=world_size,
            process_index=rank,
            is_fsdp2=True,
            is_main_process=rank == 0,
            wait_for_everyone=dist.barrier,
        )

        with patch("accelerate.utils.fsdp_utils.logger.info"):
            checkpoint_kwargs = {"adapter_only": True}
            if omit_placeholder:
                checkpoint_kwargs["excluded_parameter_names"] = ("placeholder",)
            save_fsdp_model(plugin, accelerator, model, output_dir, **checkpoint_kwargs)
            dist.barrier()

            checkpoint_path = os.path.join(output_dir, "pytorch_model_fsdp.bin")
            checkpoint_error = None
            if rank == 0:
                try:
                    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    assert set(saved) == {"adapter"}
                    torch.testing.assert_close(saved["adapter"], expected_adapter.cpu())
                    assert os.path.getsize(checkpoint_path) < 64 * 1024
                except Exception as error:
                    checkpoint_error = f"{type(error).__name__}: {error}"
            checkpoint_errors = [None] * world_size
            dist.all_gather_object(checkpoint_errors, checkpoint_error)
            assert not any(checkpoint_errors), checkpoint_errors

            with torch.no_grad():
                model.adapter.to_local().fill_(-7)
                model.frozen.to_local().fill_(rank + 20)
            expected_mutated_base = model.frozen.full_tensor().detach().clone()
            expected_placeholder = (
                model.placeholder.detach().clone()
                if omit_placeholder
                else model.placeholder.full_tensor().detach().clone()
            )
            expected_placeholder_layout = (
                model.placeholder.shape,
                model.placeholder.stride(),
                model.placeholder.untyped_storage().nbytes(),
            )

            load_fsdp_model(plugin, accelerator, model, output_dir, **checkpoint_kwargs)
            torch.testing.assert_close(model.adapter.full_tensor(), expected_adapter)
            torch.testing.assert_close(model.frozen.full_tensor(), expected_mutated_base)
            restored_placeholder = model.placeholder if omit_placeholder else model.placeholder.full_tensor()
            torch.testing.assert_close(restored_placeholder, expected_placeholder)
            if omit_placeholder:
                assert (
                    restored_placeholder.shape,
                    restored_placeholder.stride(),
                    restored_placeholder.untyped_storage().nbytes(),
                ) == expected_placeholder_layout
            assert not model.placeholder.requires_grad

            dist.barrier()
            if rank == 0:
                torch.save({}, checkpoint_path)
            dist.barrier()
            with pytest.raises(RuntimeError, match="rank 0: adapter state keys do not match"):
                load_fsdp_model(plugin, accelerator, model, output_dir, **checkpoint_kwargs)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("is_main_process", [True, False])
def test_fsdp2_load_full_state_dict_materializes_ordinary_parameters_and_buffers(is_main_process):
    model = ModelWithPersistentBuffer()
    full_state_dict = model.state_dict() if is_main_process else {}
    model.to(torch.device("meta"))
    accelerator = SimpleNamespace(is_main_process=is_main_process, device=torch.device("cpu"))

    received_tensors = iter((torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))

    def broadcast(tensor, src, group):
        expected = next(received_tensors)
        if not is_main_process:
            tensor.copy_(expected)

    with patch("torch.distributed.broadcast", side_effect=broadcast) as mock_broadcast:
        fsdp2_load_full_state_dict(accelerator, model, full_state_dict)

    assert mock_broadcast.call_count == 2
    torch.testing.assert_close(model.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.router_bias, torch.tensor([3.0, 4.0]))
    assert model.weight.device.type == "cpu"
    assert model.router_bias.device.type == "cpu"


@pytest.mark.parametrize("is_main_process", [True, False])
def test_fsdp2_load_full_state_dict_leaves_only_registered_parameters_rank_local(is_main_process):
    model = AdapterStateModel()
    full_state_dict = model.state_dict() if is_main_process else {}
    model.to(torch.device("meta"))
    accelerator = SimpleNamespace(is_main_process=is_main_process, device=torch.device("cpu"))
    received_tensors = iter((torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])))

    def broadcast(tensor, src, group):
        expected = next(received_tensors)
        if not is_main_process:
            tensor.copy_(expected)

    with patch("torch.distributed.broadcast", side_effect=broadcast) as mock_broadcast:
        fsdp2_load_full_state_dict(
            accelerator,
            model,
            full_state_dict,
            rank_local_parameter_names=("placeholder",),
        )

    assert mock_broadcast.call_count == 2
    torch.testing.assert_close(model.frozen, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.adapter, torch.tensor([3.0, 4.0]))
    assert model.placeholder.device.type == ("cpu" if is_main_process else "meta")


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


def test_fsdp2_staged_optimizer_rejects_stale_model_parameter_even_when_mixed():
    source_parameter = torch.nn.Parameter(torch.ones(1))
    prepared_model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW([source_parameter, *prepared_model.parameters()])
    accelerator = SimpleNamespace(
        _models=[prepared_model],
        _fsdp2_source_parameter_refs=(weakref.ref(source_parameter),),
    )

    with pytest.raises(ValueError, match="before it was prepared"):
        Accelerator._validate_fsdp2_prepare_inputs(accelerator, [], [optimizer])


def test_fsdp2_staged_optimizer_ignores_released_source_parameter_identity():
    released_source_parameter = torch.nn.Parameter(torch.ones(1))
    released_source_ref = weakref.ref(released_source_parameter)
    del released_source_parameter
    gc.collect()
    assert released_source_ref() is None

    prepared_model = torch.nn.Linear(1, 1)
    external_parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([*prepared_model.parameters(), external_parameter])
    accelerator = SimpleNamespace(
        _models=[prepared_model],
        _fsdp2_source_parameter_refs=(released_source_ref,),
    )

    Accelerator._validate_fsdp2_prepare_inputs(accelerator, [], [optimizer])


def test_fsdp2_staged_optimizer_validates_each_optimizer():
    prepared_model = torch.nn.Linear(1, 1)
    valid_optimizer = torch.optim.AdamW(prepared_model.parameters())
    unrelated_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    accelerator = SimpleNamespace(_models=[prepared_model], _fsdp2_source_parameter_refs=())

    with pytest.raises(ValueError, match="optimizer 1 does not reference"):
        Accelerator._validate_fsdp2_prepare_inputs(
            accelerator,
            [],
            [valid_optimizer, unrelated_optimizer],
        )


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


def test_fsdp2_rank_local_registry_validates_name_and_identity():
    accelerator = SimpleNamespace(is_fsdp2=True)
    model = AdapterStateModel()

    Accelerator.register_fsdp2_rank_local_parameters(accelerator, model, ("placeholder",))
    registered = Accelerator._get_fsdp2_rank_local_parameters(accelerator, model)
    assert registered == ("placeholder",)

    registered_parameter_ref = weakref.ref(model.placeholder)
    model.placeholder = torch.nn.Parameter(torch.zeros(2))
    assert registered_parameter_ref() is None
    with pytest.raises(RuntimeError, match="changed before model preparation"):
        Accelerator._get_fsdp2_rank_local_parameters(accelerator, model)

    model = AdapterStateModel()
    with pytest.raises(ValueError, match="not found"):
        Accelerator.register_fsdp2_rank_local_parameters(accelerator, model, ("missing",))
    with pytest.raises(ValueError, match="duplicates"):
        Accelerator.register_fsdp2_rank_local_parameters(accelerator, model, ("placeholder", "placeholder"))


def test_fsdp2_prepare_resolves_registered_parameter_identity_after_meta_move():
    from torch.distributed.fsdp import OffloadPolicy

    plugin = SimpleNamespace(
        activation_checkpointing=False,
        auto_wrap_policy=None,
        cpu_offload=OffloadPolicy(),
        cpu_ram_efficient_loading=True,
        ignored_modules=None,
        mixed_precision_policy=None,
        reshard_after_forward=True,
        set_auto_wrap_policy=lambda model: None,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=True,
        mixed_precision="no",
        parallelism_config=None,
        state=SimpleNamespace(fsdp_plugin=plugin),
    )
    model = AdapterStateModel()
    registered_parameter = model.placeholder
    captured_kwargs = {}

    def fully_shard(module, **kwargs):
        captured_kwargs.update(kwargs)

    with (
        patch("torch.distributed.fsdp.fully_shard", side_effect=fully_shard),
        patch("accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict") as mock_load,
    ):
        fsdp2_prepare_model(
            accelerator,
            model,
            rank_local_parameter_names=("placeholder",),
        )

    assert model.placeholder is not registered_parameter
    assert captured_kwargs["ignored_params"] == {model.placeholder}
    assert mock_load.call_args.kwargs["rank_local_parameter_names"] == ("placeholder",)


def test_fsdp2_prepare_unions_all_ignored_parameter_sources():
    from torch.distributed.fsdp import OffloadPolicy

    class Params4bit(torch.nn.Parameter):
        pass

    class CombinedIgnoredModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.quantized = Params4bit(torch.ones(2, dtype=torch.uint8), requires_grad=False)
            self.ignored_module = torch.nn.Linear(2, 2)
            self.rank_local = torch.nn.Parameter(torch.ones(2))

    model = CombinedIgnoredModel()
    plugin = SimpleNamespace(
        auto_wrap_policy=None,
        cpu_offload=OffloadPolicy(),
        cpu_ram_efficient_loading=False,
        ignored_modules=(model.ignored_module,),
        mixed_precision_policy=None,
        reshard_after_forward=True,
        set_auto_wrap_policy=lambda model: None,
    )
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=False,
        mixed_precision="no",
        parallelism_config=None,
        state=SimpleNamespace(fsdp_plugin=plugin),
    )
    captured_kwargs = {}

    def fully_shard(module, **kwargs):
        captured_kwargs.update(kwargs)

    with patch("torch.distributed.fsdp.fully_shard", side_effect=fully_shard):
        fsdp2_prepare_model(
            accelerator,
            model,
            rank_local_parameter_names=("rank_local",),
        )

    assert captured_kwargs["ignored_params"] == {
        model.quantized,
        model.ignored_module.weight,
        model.ignored_module.bias,
        model.rank_local,
    }


def test_fsdp2_joint_prepare_preserves_external_optimizer_parameter_identity():
    class JointPrepareHarness:
        def __init__(self):
            self.state = SimpleNamespace(
                fsdp_plugin=SimpleNamespace(
                    activation_checkpointing=False,
                    set_auto_wrap_policy=lambda model: None,
                ),
                dynamo_plugin=SimpleNamespace(backend=DynamoBackend.NO),
            )
            self._models = []
            self.fp8_backend = FP8BackendType.NO
            self._fsdp2_source_parameter_refs = ()

        def _prepare_one(self, obj, **kwargs):
            return obj

        def _get_named_parameters(self, *objects, drop_refs=False):
            return Accelerator._get_named_parameters(self, *objects, drop_refs=drop_refs)

        def _get_fsdp2_rank_local_parameters(self, model):
            return {}

    accelerator = JointPrepareHarness()
    model = torch.nn.Linear(2, 2)
    external_parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([model.weight, model.bias, external_parameter])

    with patch("accelerate.accelerator.fsdp2_prepare_model", return_value=model):
        _, prepared_optimizer = Accelerator._prepare_fsdp2(accelerator, model, optimizer)

    prepared_parameters = prepared_optimizer.param_groups[0]["params"]
    assert prepared_parameters[0] is model.weight
    assert prepared_parameters[1] is model.bias
    assert prepared_parameters[2] is external_parameter


def test_fsdp2_optimizer_switch_rejects_model_owned_mapping_miss():
    model_owned_parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([model_owned_parameter])

    with pytest.raises(KeyError, match="model-owned optimizer parameter"):
        fsdp2_switch_optimizer_parameters(
            optimizer,
            {},
            model_owned_parameter_ids=(id(model_owned_parameter),),
        )


def test_rank_local_gradient_clipping_in_non_distributed_training():
    model_parameter = torch.nn.Parameter(torch.ones(1))
    model_parameter.grad = torch.tensor([3.0])
    external_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    external_parameter.grad = torch.tensor([4.0], dtype=torch.bfloat16)
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO,
        is_fsdp2=False,
        unscale_gradients=lambda: None,
    )

    total_norm = Accelerator.clip_grad_norm_(
        accelerator,
        (model_parameter,),
        max_norm=1.0,
        rank_local_parameters=(external_parameter,),
    )

    torch.testing.assert_close(total_norm, torch.tensor(5.0))
    torch.testing.assert_close(model_parameter.grad, torch.tensor([0.6]))
    torch.testing.assert_close(external_parameter.grad.float(), torch.tensor([0.8]), atol=0.005, rtol=0)


def test_rank_local_gradient_clipping_handles_empty_and_none_gradients():
    parameter_without_gradient = torch.nn.Parameter(torch.ones(1))
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO,
        is_fsdp2=False,
        unscale_gradients=lambda: None,
    )

    total_norm = Accelerator.clip_grad_norm_(
        accelerator,
        (parameter_without_gradient,),
        max_norm=1.0,
        rank_local_parameters=(),
    )

    torch.testing.assert_close(total_norm, torch.tensor(0.0))
    assert parameter_without_gradient.grad is None


@pytest.mark.parametrize(
    ("rank_local_parameters", "norm_type", "error_message"),
    [
        ("duplicate", 2, "must not contain duplicates"),
        ("overlap", 2, "must be identity-disjoint"),
        ("empty", 1, "supports only `norm_type=2`"),
    ],
)
def test_rank_local_gradient_clipping_validates_inputs(rank_local_parameters, norm_type, error_message):
    parameter = torch.nn.Parameter(torch.ones(1))
    extras = {
        "duplicate": (parameter, parameter),
        "overlap": (parameter,),
        "empty": (),
    }[rank_local_parameters]
    model_parameters = () if rank_local_parameters == "duplicate" else (parameter,)
    accelerator = SimpleNamespace(
        distributed_type=DistributedType.NO,
        is_fsdp2=False,
        unscale_gradients=lambda: None,
    )

    with pytest.raises(ValueError, match=error_message):
        Accelerator.clip_grad_norm_(
            accelerator,
            model_parameters,
            max_norm=1.0,
            norm_type=norm_type,
            rank_local_parameters=extras,
        )


def test_fsdp2_gradient_clip_collective_device_follows_backend():
    with patch("torch.distributed.get_backend", return_value="gloo"):
        assert Accelerator._fsdp2_grad_clip_collective_device(torch.device("cuda:1")) == torch.device("cpu")
    with patch("torch.distributed.get_backend", return_value="nccl"):
        assert Accelerator._fsdp2_grad_clip_collective_device(torch.device("cuda:1")) == torch.device("cuda:1")


def test_rank_local_gradient_clipping_rejects_non_fsdp2_distributed_training():
    accelerator = SimpleNamespace(distributed_type=DistributedType.MULTI_CPU, is_fsdp2=False)

    with pytest.raises(RuntimeError, match="only in non-distributed training or with FSDP2"):
        Accelerator.clip_grad_norm_(
            accelerator,
            (),
            max_norm=1.0,
            rank_local_parameters=(),
        )


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires torch.distributed gloo")
def test_fsdp2_state_dict_contract_is_symmetric_across_two_ranks():
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous_path = f"{temporary_directory}/rendezvous"
        mp.spawn(_run_two_rank_state_dict_checks, args=(2, rendezvous_path), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires torch.distributed gloo")
def test_fsdp2_combined_gradient_clipping_is_symmetric_across_two_ranks():
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous_path = f"{temporary_directory}/rendezvous"
        mp.spawn(_run_two_rank_gradient_clip_checks, args=(2, rendezvous_path), nprocs=2, join=True)


@pytest.mark.parametrize("omit_placeholder", [False, True])
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires torch.distributed gloo")
def test_fsdp2_adapter_checkpoint_round_trip_is_exact_across_two_ranks(omit_placeholder):
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous_path = f"{temporary_directory}/rendezvous"
        output_dir = f"{temporary_directory}/checkpoint"
        mp.spawn(
            _run_two_rank_adapter_checkpoint_checks,
            args=(2, rendezvous_path, output_dir, omit_placeholder),
            nprocs=2,
            join=True,
        )

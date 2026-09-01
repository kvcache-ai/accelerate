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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from accelerate import Accelerator
from accelerate.utils import DistributedType


class _PartialPlacement:
    def is_replicate(self):
        return False


class _PartialNorm:
    placements = (_PartialPlacement(),)

    def __init__(self):
        self.full_tensor_called = False

    def full_tensor(self):
        self.full_tensor_called = True
        return torch.tensor(3.0)

    def to_local(self):
        raise AssertionError("a partial norm must be reduced before reading its local tensor")


class _GradientClipModel(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, device=f"cuda:{rank}"))


def _run_two_rank_nccl_gradient_clip(rank, world_size, rendezvous_path):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torch.distributed.fsdp import fully_shard

        model = _GradientClipModel(rank)
        fully_shard(model)
        model.weight.grad = torch.zeros_like(model.weight)
        model.weight.grad.to_local().fill_(3.0 if rank == 0 else 0.0)

        external_parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        external_parameter.grad = torch.tensor([4.0], dtype=torch.bfloat16)
        rank_local_parameters = (external_parameter,) if rank == 0 else ()
        accelerator = SimpleNamespace(
            device=torch.device(f"cuda:{rank}"),
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

        torch.testing.assert_close(total_norm, torch.tensor(5.0, device=f"cuda:{rank}"))
        coefficient = 1.0 / (5.0 + 1e-6)
        torch.testing.assert_close(
            model.weight.grad.to_local(),
            torch.tensor([3.0 * coefficient if rank == 0 else 0.0], device=f"cuda:{rank}"),
        )
        if rank == 0:
            torch.testing.assert_close(
                external_parameter.grad.float(),
                torch.tensor([4.0 * coefficient]),
                atol=0.005,
                rtol=0,
            )
    finally:
        dist.destroy_process_group()


def test_fsdp2_partial_gradient_norm_is_materialized_before_local_access():
    partial_norm = _PartialNorm()

    result = Accelerator._materialize_fsdp2_grad_norm(partial_norm)

    torch.testing.assert_close(result, torch.tensor(3.0))
    assert partial_norm.full_tensor_called


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and NCCL",
)
def test_fsdp2_combined_gradient_clipping_with_partial_norm_on_nccl():
    with tempfile.TemporaryDirectory() as temporary_directory:
        rendezvous_path = os.path.join(temporary_directory, "rendezvous")
        mp.spawn(_run_two_rank_nccl_gradient_clip, args=(2, rendezvous_path), nprocs=2, join=True)

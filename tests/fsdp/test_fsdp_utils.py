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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from accelerate.utils.fsdp_utils import fsdp2_load_full_state_dict


class ModelWithPersistentBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("router_bias", torch.tensor([3.0, 4.0]))


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

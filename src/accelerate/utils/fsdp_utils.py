# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import copy
import functools
import os
import re
import shutil
import warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Union

import torch

from ..logging import get_logger
from .constants import FSDP_MODEL_NAME, OPTIMIZER_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import get_module_class_from_name
from .modeling import get_non_persistent_buffers, is_peft_model
from .other import get_module_children_bottom_up, is_compiled_module, save
from .versions import is_torch_version


logger = get_logger(__name__)


def enable_fsdp_ram_efficient_loading():
    """
    Enables RAM efficient loading of Hugging Face models for FSDP in the environment.
    """
    # Sets values for `transformers.modeling_utils.is_fsdp_enabled`
    if "ACCELERATE_USE_FSDP" not in os.environ:
        os.environ["ACCELERATE_USE_FSDP"] = "True"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"


def disable_fsdp_ram_efficient_loading():
    """
    Disables RAM efficient loading of Hugging Face models for FSDP in the environment.
    """
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "False"


def _get_model_state_dict(model, adapter_only=False, sd_options=None):
    if adapter_only and is_peft_model(model):
        if sd_options is not None:
            from torch.distributed.checkpoint.state_dict import get_model_state_dict

            adapter_options = copy.copy(sd_options)
            adapter_options.ignore_frozen_params = True
            state_dict = get_model_state_dict(model, options=adapter_options)
            adapter_names = _fsdp2_adapter_parameter_names(model)
            state_dict = {name: value for name, value in state_dict.items() if name in adapter_names}
            _validate_fsdp2_adapter_state_dict(model, state_dict, adapter_options)
            return state_dict

        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(model, adapter_name=model.active_adapter)

    # Invariant: `sd_options` is not None only for FSDP2
    if sd_options is not None:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        return get_model_state_dict(model, options=sd_options)
    else:
        return model.state_dict()


def _set_model_state_dict(model, state_dict, adapter_only=False, sd_options=None):
    if adapter_only and is_peft_model(model):
        if sd_options is not None:
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            adapter_options = copy.copy(sd_options)
            adapter_options.ignore_frozen_params = True
            adapter_options.strict = False
            _validate_fsdp2_adapter_state_dict(model, state_dict, adapter_options)
            return set_model_state_dict(model, state_dict, options=adapter_options)

        from peft import set_peft_model_state_dict

        return set_peft_model_state_dict(model, state_dict, adapter_name=model.active_adapter)

    # Invariant: `sd_options` is not None only for FSDP2
    if sd_options is not None:
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        return set_model_state_dict(model, state_dict, options=sd_options)
    else:
        return model.load_state_dict(state_dict)


def _prepare_sd_options(fsdp_plugin):
    sd_options = None

    # we use this only for FSDP2, as it requires torch >= 2.6.0 and this api requires torch >= 2.2.0
    if fsdp_plugin.fsdp_version == 2:
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        sd_options = StateDictOptions(
            full_state_dict=fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT,
            cpu_offload=getattr(fsdp_plugin.state_dict_config, "offload_to_cpu", False),
            broadcast_from_rank0=getattr(fsdp_plugin.state_dict_config, "rank0_only", False),
        )

    return sd_options


def _synchronize_state_dict_preflight(local_error):
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return (local_error,) if local_error is not None else ()

    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, local_error)
    return tuple(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)


def _fsdp2_adapter_parameter_names(model):
    return {name for name, parameter in model.named_parameters(remove_duplicate=False) if parameter.requires_grad}


def _validate_fsdp2_adapter_state_dict(model, state_dict, options):
    expected_names = _fsdp2_adapter_parameter_names(model)
    rank0_broadcast = (
        options.full_state_dict
        and options.broadcast_from_rank0
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    should_have_state = not rank0_broadcast or torch.distributed.get_rank() == 0
    local_error = None
    if should_have_state and set(state_dict) != expected_names:
        local_error = (
            "adapter state keys do not match the trainable model parameters: "
            f"missing={sorted(expected_names.difference(state_dict))}, "
            f"unexpected={sorted(set(state_dict).difference(expected_names))}"
        )
    elif not should_have_state and state_dict:
        local_error = "non-owner rank received a full adapter state dict"

    errors = _synchronize_state_dict_preflight(local_error)
    if errors:
        raise RuntimeError(f"FSDP2 adapter state-dict preflight failed: {'; '.join(errors)}")


def _get_fsdp2_model_state_dict(model, adapter_only=False, excluded_parameter_names=()):
    """Collect an FSDP2 full state dict without gathering frozen parameters in adapter-only mode."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    local_error = None
    excluded_names = ()
    named_parameters = {}
    try:
        if not isinstance(adapter_only, bool):
            raise TypeError("`adapter_only` must be a boolean.")
        if isinstance(excluded_parameter_names, str):
            raise TypeError("`excluded_parameter_names` must be an iterable of parameter names, not a string.")

        excluded_names = tuple(sorted(set(excluded_parameter_names)))
        if any(not isinstance(name, str) for name in excluded_names):
            raise TypeError("`excluded_parameter_names` must contain only strings.")

        named_parameters = dict(model.named_parameters(remove_duplicate=False))
        missing_names = sorted(set(excluded_names).difference(named_parameters))
        if missing_names:
            raise ValueError(f"Excluded parameter names were not found in the model: {missing_names}")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"

    errors = _synchronize_state_dict_preflight(local_error)
    if errors:
        raise RuntimeError(f"FSDP2 state-dict preflight failed: {'; '.join(errors)}")

    signature = (bool(adapter_only), excluded_names)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        signatures = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(signatures, signature)
        if any(candidate != signature for candidate in signatures):
            raise RuntimeError(
                "FSDP2 state-dict selection must use identical `adapter_only` and "
                "`excluded_parameter_names` values on every rank."
            )

    included_names = {
        name for name, parameter in named_parameters.items() if parameter.requires_grad and name not in excluded_names
    }
    excluded_parameters = {id(named_parameters[name]): named_parameters[name] for name in excluded_names}
    original_requires_grad = {identity: parameter.requires_grad for identity, parameter in excluded_parameters.items()}

    mutation_error = None
    if adapter_only:
        try:
            # DCP's frozen-parameter filter expects every frozen parameter to be present in ``model.state_dict()``.
            # Some integrations intentionally omit placeholder parameters, so make explicitly excluded placeholders
            # visible to the filter and remove them from the result below.
            for parameter in excluded_parameters.values():
                parameter.requires_grad_(True)
        except Exception as error:
            mutation_error = f"{type(error).__name__}: {error}"

    errors = _synchronize_state_dict_preflight(mutation_error)
    if errors:
        for identity, parameter in excluded_parameters.items():
            parameter.requires_grad_(original_requires_grad[identity])
        raise RuntimeError(f"FSDP2 state-dict preflight failed while preparing exclusions: {'; '.join(errors)}")

    try:
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                ignore_frozen_params=adapter_only,
                broadcast_from_rank0=True,
            ),
        )
    finally:
        for identity, parameter in excluded_parameters.items():
            parameter.requires_grad_(original_requires_grad[identity])

    if adapter_only:
        return {name: value for name, value in state_dict.items() if name in included_names}

    for name in excluded_names:
        state_dict.pop(name, None)
    return state_dict


def save_fsdp_model(fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    os.makedirs(output_dir, exist_ok=True)
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)

    with ctx:
        state_dict = _get_model_state_dict(model, adapter_only=adapter_only, sd_options=sd_options)
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                logger.info(f"Saving model to {output_model_file}")
                torch.save(state_dict, output_model_file)
                logger.info(f"Model saved to {output_model_file}")
        # Invariant: `LOCAL_STATE_DICT` is never possible with `FSDP2`
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            output_model_file = os.path.join(output_dir, weights_name)
            logger.info(f"Saving model to {output_model_file}")
            torch.save(state_dict, output_model_file)
            logger.info(f"Model saved to {output_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{model_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving model to {ckpt_dir}")
            state_dict = {"model": state_dict}

            dist_cp.save(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Model saved to {ckpt_dir}")


def load_fsdp_model(fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
        # FSDP raises error when single GPU is used with `offload_to_cpu=True` for FULL_STATE_DICT
        # so, only enable it when num_processes>1
        is_multi_process = accelerator.num_processes > 1
        fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
        fsdp_plugin.state_dict_config.rank0_only = is_multi_process

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)
    with ctx:
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) is not FSDP and accelerator.process_index != 0 and not accelerator.is_fsdp2:
                if not fsdp_plugin.sync_module_states and fsdp_plugin.fsdp_version == 1:
                    raise ValueError(
                        "Set the `sync_module_states` flag to `True` so that model states are synced across processes when "
                        "initializing FSDP object"
                    )
                return
            weights_name = f"{FSDP_MODEL_NAME}.bin" if model_index == 0 else f"{FSDP_MODEL_NAME}_{model_index}.bin"
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            # we want an empty state dict for FSDP2 as we use `broadcast_from_rank0`
            load_model = not accelerator.is_fsdp2 or accelerator.is_main_process
            if load_model:
                state_dict = torch.load(input_model_file, weights_only=True)
            else:
                state_dict = {}
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.LOCAL_STATE_DICT:
            weights_name = (
                f"{FSDP_MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{FSDP_MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            logger.info(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file, weights_only=True)
            logger.info(f"Model loaded from {input_model_file}")
        elif fsdp_plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT:
            ckpt_dir = (
                os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{model_index}")
                if f"{FSDP_MODEL_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading model from {ckpt_dir}")
            state_dict = {"model": _get_model_state_dict(model, adapter_only=adapter_only, sd_options=sd_options)}
            dist_cp.load(
                state_dict=state_dict,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
                planner=DefaultLoadPlanner(),
            )
            state_dict = state_dict["model"]
            logger.info(f"Model loaded from {ckpt_dir}")

        load_result = _set_model_state_dict(model, state_dict, adapter_only=adapter_only, sd_options=sd_options)
    return load_result


def save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    os.makedirs(output_dir, exist_ok=True)

    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )

    sd_options = _prepare_sd_options(fsdp_plugin)

    with ctx:
        if fsdp_plugin.fsdp_version == 2:
            from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

            optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)
        else:
            optim_state = FSDP.optim_state_dict(model, optimizer)

        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            if accelerator.process_index == 0:
                optim_state_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                output_optimizer_file = os.path.join(output_dir, optim_state_name)
                logger.info(f"Saving Optimizer state to {output_optimizer_file}")
                torch.save(optim_state, output_optimizer_file)
                logger.info(f"Optimizer state saved in {output_optimizer_file}")
        else:
            ckpt_dir = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
            os.makedirs(ckpt_dir, exist_ok=True)
            logger.info(f"Saving Optimizer state to {ckpt_dir}")
            dist_cp.save(
                state_dict={"optimizer": optim_state},
                storage_writer=dist_cp.FileSystemWriter(ckpt_dir),
                planner=DefaultSavePlanner(),
            )
            logger.info(f"Optimizer state saved in {ckpt_dir}")


def load_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, input_dir, optimizer_index=0, adapter_only=False):
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    accelerator.wait_for_everyone()
    ctx = (
        FSDP.state_dict_type(
            model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
        )
        if fsdp_plugin.fsdp_version == 1
        else nullcontext()
    )
    sd_options = _prepare_sd_options(fsdp_plugin)
    with ctx:
        if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
            optim_state = None
            if accelerator.process_index == 0 or not fsdp_plugin.optim_state_dict_config.rank0_only:
                optimizer_name = (
                    f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
                )
                input_optimizer_file = os.path.join(input_dir, optimizer_name)
                logger.info(f"Loading Optimizer state from {input_optimizer_file}")
                optim_state = torch.load(input_optimizer_file, weights_only=True)
                logger.info(f"Optimizer state loaded from {input_optimizer_file}")
        else:
            ckpt_dir = (
                os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
                if f"{OPTIMIZER_NAME}" not in input_dir
                else input_dir
            )
            logger.info(f"Loading Optimizer from {ckpt_dir}")
            if fsdp_plugin.fsdp_version == 2:
                from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

                optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)
            else:
                optim_state = FSDP.optim_state_dict(model, optimizer)
            optim_state = {"optimizer": optim_state}
            dist_cp.load(
                optim_state,
                checkpoint_id=ckpt_dir,
                storage_reader=dist_cp.FileSystemReader(ckpt_dir),
            )
            optim_state = optim_state["optimizer"]
            logger.info(f"Optimizer loaded from {ckpt_dir}")

        if fsdp_plugin.fsdp_version == 1:
            flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=optimizer, optim_state_dict=optim_state)
            optimizer.load_state_dict(flattened_osd)
        else:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

            set_optimizer_state_dict(model, optimizer, optim_state, options=sd_options)


def _distributed_checkpoint_to_merged_weights(checkpoint_dir: str, save_path: str, safe_serialization: bool = True):
    """
    Passthrough to `torch.distributed.checkpoint.format_utils.dcp_to_torch_save`

    Will save under `save_path` as either `model.safetensors` or `pytorch_model.bin`.
    """
    # Note: We import here to reduce import time from general modules, and isolate outside dependencies
    import torch.distributed.checkpoint as dist_cp
    import torch.distributed.checkpoint.format_utils as dist_cp_format_utils

    state_dict = {}
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    dist_cp_format_utils._load_state_dict(
        state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=dist_cp_format_utils._EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    save_path = save_path / SAFE_WEIGHTS_NAME if safe_serialization else save_path / WEIGHTS_NAME

    # To handle if state is a dict like {model: {...}}
    if len(state_dict.keys()) == 1:
        state_dict = state_dict[list(state_dict)[0]]
    save(state_dict, save_path, safe_serialization=safe_serialization)
    return save_path


def merge_fsdp_weights(
    checkpoint_dir: str, output_path: str, safe_serialization: bool = True, remove_checkpoint_dir: bool = False
):
    """
    Merge the weights from sharded FSDP model checkpoints into a single combined checkpoint. Should be used if
    `SHARDED_STATE_DICT` was used for the model. Weights will be saved to `{output_path}/model.safetensors` if
    `safe_serialization` else `pytorch_model.bin`.

    Note: this is a CPU-bound process.

    Args:
        checkpoint_dir (`str`):
            The directory containing the FSDP checkpoints (can be either the model or optimizer).
        output_path (`str`):
            The path to save the merged checkpoint.
        safe_serialization (`bool`, *optional*, defaults to `True`):
            Whether to save the merged weights with safetensors (recommended).
        remove_checkpoint_dir (`bool`, *optional*, defaults to `False`):
            Whether to remove the checkpoint directory after merging.
    """
    checkpoint_dir = Path(checkpoint_dir)
    from accelerate.state import PartialState

    if not is_torch_version(">=", "2.3.0"):
        raise ValueError("`merge_fsdp_weights` requires PyTorch >= 2.3.0`")

    # Verify that the checkpoint directory exists
    if not checkpoint_dir.exists():
        model_path_exists = (checkpoint_dir / "pytorch_model_fsdp_0").exists()
        optimizer_path_exists = (checkpoint_dir / "optimizer_0").exists()
        err = f"Tried to load from {checkpoint_dir} but couldn't find a valid metadata file."
        if model_path_exists and optimizer_path_exists:
            err += " However, potential model and optimizer checkpoint directories exist."
            err += f"Please pass in either {checkpoint_dir}/pytorch_model_fsdp_0 or {checkpoint_dir}/optimizer_0"
            err += "instead."
        elif model_path_exists:
            err += " However, a potential model checkpoint directory exists."
            err += f"Please try passing in {checkpoint_dir}/pytorch_model_fsdp_0 instead."
        elif optimizer_path_exists:
            err += " However, a potential optimizer checkpoint directory exists."
            err += f"Please try passing in {checkpoint_dir}/optimizer_0 instead."
        raise ValueError(err)

    # To setup `save` to work
    state = PartialState()
    if state.is_main_process:
        logger.info(f"Merging FSDP weights from {checkpoint_dir}")
        save_path = _distributed_checkpoint_to_merged_weights(checkpoint_dir, output_path, safe_serialization)
        logger.info(f"Successfully merged FSDP weights and saved to {save_path}")
        if remove_checkpoint_dir:
            logger.info(f"Removing old checkpoint directory {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
    state.wait_for_everyone()


def ensure_weights_retied(param_init_fn, model: torch.nn.Module, device: torch.device):
    _tied_names = getattr(model, "_tied_weights_keys", None)
    if not _tied_names:
        # if no tied names just passthrough
        return param_init_fn

    # get map of parameter instances to params.
    # - needed for replacement later
    _tied_params = {}
    for name in _tied_names:
        name = name.split(".")
        name, param_name = ".".join(name[:-1]), name[-1]
        mod = model.get_submodule(name)
        param = getattr(mod, param_name)

        _tied_params[id(param)] = None  # placeholder for the param first

    # build param_init_fn for the case with tied params
    def param_init_fn_tied_param(module: torch.nn.Module):
        # track which params to tie
        # - usually only 1, but for completeness consider > 1
        params_to_tie = defaultdict(list)
        for n, param in module.named_parameters(recurse=False):
            if id(param) in _tied_params:
                params_to_tie[id(param)].append(n)

        # call the param init fn, which potentially re-allocates the
        # parameters
        module = param_init_fn(module)

        # search the parameters again and tie them up again
        for id_key, _param_names in params_to_tie.items():
            for param_name in _param_names:
                param = _tied_params[id_key]
                if param is None:
                    # everything will be tied to the first time the
                    # param is observed
                    _tied_params[id_key] = getattr(module, param_name)
                else:
                    setattr(module, param_name, param)  # tie

        return module

    return param_init_fn_tied_param


def fsdp2_load_full_state_dict(
    accelerator,
    model: torch.nn.Module,
    full_sd: dict,
    cpu_offload: bool = False,
    rank_local_parameter_names: Iterable[str] = (),
):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can only be on rank 0
        cpu_offload (`bool`, defaults to `False`):
            If True, move sharded parameters to CPU after distribution. Required when FSDP CPU offloading is enabled.
        rank_local_parameter_names (`Iterable[str]`, defaults to `()`):
            Fully qualified parameter names owned by an external integration. These parameters are not broadcast or
            materialized by this function.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, distribute_tensor

    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    local_error = None
    try:
        if isinstance(rank_local_parameter_names, str):
            raise TypeError("`rank_local_parameter_names` must be an iterable of parameter names, not a string.")
        rank_local_parameter_names = tuple(rank_local_parameter_names)
        if any(not isinstance(name, str) or not name for name in rank_local_parameter_names):
            raise TypeError("`rank_local_parameter_names` must contain only non-empty strings.")
        if len(rank_local_parameter_names) != len(set(rank_local_parameter_names)):
            raise ValueError("`rank_local_parameter_names` must not contain duplicates.")
        named_parameters = fsdp2_canonicalize_names(dict(model.named_parameters()))
        missing_rank_local_names = sorted(set(rank_local_parameter_names).difference(named_parameters))
        del named_parameters
        if missing_rank_local_names:
            raise ValueError(f"Rank-local parameter names were not found in the model: {missing_rank_local_names}")
    except Exception as error:
        local_error = error
        rank_local_parameter_names = ()

    if dist.is_available() and dist.is_initialized():
        local_error_message = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
        payloads = [None] * dist.get_world_size()
        dist.all_gather_object(payloads, (rank_local_parameter_names, local_error_message))
        errors = [f"rank {rank}: {payload[1]}" for rank, payload in enumerate(payloads) if payload[1] is not None]
        if errors:
            raise RuntimeError(f"FSDP2 full-state rank-local preflight failed: {'; '.join(errors)}")
        if any(payload[0] != payloads[0][0] for payload in payloads[1:]):
            raise RuntimeError("FSDP2 full-state rank-local parameter names must be identical on every rank.")
    elif local_error is not None:
        raise local_error
    rank_local_parameter_names = set(rank_local_parameter_names)

    def _canonicalize_name(name):
        return next(iter(fsdp2_canonicalize_names({name: None})))

    # Rank 0 distributes the full state dict to other ranks
    def _infer_parameter_dtype(model, param_name, empty_param):
        try:
            old_param = model.get_parameter(param_name)
        except AttributeError:
            try:
                old_param = model.get_buffer(param_name)
            except AttributeError:
                # Need this for LoRA, as some state-dict values are plain tensor attributes.
                if "." in param_name:
                    base_param_name, local_param_name = param_name.rsplit(".", 1)
                    submodule = model.get_submodule(base_param_name)
                else:
                    local_param_name = param_name
                    submodule = model
                old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    def _is_dtensor(param):
        """Check whether a state-dict value is managed by FSDP2."""
        return hasattr(param, "device_mesh") and param.device_mesh is not None

    if accelerator.is_main_process:
        for param_name, sharded_param in meta_sharded_sd.items():
            if param_name not in full_sd:
                raise KeyError(
                    f"Parameter '{param_name}' found in sharded model state dict but missing from full state dict. "
                    f"Full state dict has {len(full_sd)} keys, sharded has {len(meta_sharded_sd)} keys."
                )
            full_param = full_sd[param_name]
            if not _is_dtensor(sharded_param):
                if _canonicalize_name(param_name) in rank_local_parameter_names:
                    sharded_sd[param_name] = full_param.detach()
                    continue
                full_tensor = full_param.detach().to(device=accelerator.device, dtype=sharded_param.dtype).contiguous()
                dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
                sharded_sd[param_name] = full_tensor
                continue
            device_mesh = sharded_param.device_mesh
            full_param = full_param.detach().to(device_mesh.device_type)
            if isinstance(full_param, DTensor):
                # dist.broadcast() only supports torch.Tensor.
                # After prepare_tp(), model parameters may become DTensor.
                # To broadcast such a parameter, convert it to a local tensor first.
                full_param = full_param.to_local()
            dist.broadcast(full_param, src=0, group=dist.group.WORLD)
            sharded_tensor = distribute_tensor(full_param, device_mesh, sharded_param.placements)
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_param,
            )
            sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)
            # When CPU offloading is enabled, FSDP2's lazy_init expects parameters on CPU
            if cpu_offload:
                sharded_tensor = sharded_tensor.to("cpu")
            sharded_sd[param_name] = sharded_tensor
    # We need this else to have a matching `broadcast` for all of the ranks, else we deadlock
    else:
        for param_name, sharded_param in meta_sharded_sd.items():
            if not _is_dtensor(sharded_param):
                if _canonicalize_name(param_name) in rank_local_parameter_names:
                    sharded_sd[param_name] = sharded_param
                    continue
                full_tensor = torch.empty(sharded_param.size(), device=accelerator.device, dtype=sharded_param.dtype)
                dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
                sharded_sd[param_name] = full_tensor
                continue
            device_mesh = sharded_param.device_mesh
            full_tensor = torch.empty(sharded_param.size(), device=device_mesh.device_type, dtype=sharded_param.dtype)
            dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
            sharded_tensor = distribute_tensor(full_tensor, device_mesh, sharded_param.placements)
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_tensor,
            )
            sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)
            # When CPU offloading is enabled, FSDP2's lazy_init expects parameters on CPU
            if cpu_offload:
                sharded_tensor = sharded_tensor.to("cpu")
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    model.load_state_dict(sharded_sd, assign=True)
    return model


def fsdp2_switch_optimizer_parameters(
    optimizer: torch.optim.Optimizer, mapping: dict, model_owned_parameter_ids: Iterable[int] = ()
):
    """
    Switches the parameters of the optimizer to new ones (sharded parameters in usual case). This function modifies the
    optimizer in-place.

    Args:
        optimizer (`torch.optim.Optimizer`): Optimizer instance which contains the original model parameters
        mapping (`dict`): Mapping from the original parameter (specified by `data_ptr`) to the sharded parameter
        model_owned_parameter_ids (`Iterable[int]`, defaults to `()`): Identities of temporary model-parameter
            placeholders that must have an entry in ``mapping``. Other parameters are kept unchanged.

    Raises:
        KeyError:
            If a parameter in the optimizer couldn't be switched to its sharded version. This should never happen and
            indicates a bug. If we kept the original params instead of raising, the training wouldn't be numerically
            correct and weights wouldn't get updated.
    """
    model_owned_parameter_ids = set(model_owned_parameter_ids)
    for param_group in optimizer.param_groups:
        new_params = []
        for p in param_group["params"]:
            ptr = p.data_ptr if not callable(getattr(p, "data_ptr", None)) else p.data_ptr()
            if ptr in mapping:
                new_params.append(mapping[ptr])
            elif id(p) in model_owned_parameter_ids:
                raise KeyError("A model-owned optimizer parameter could not be mapped after FSDP2 preparation.")
            else:
                new_params.append(p)
        param_group["params"] = new_params


def fsdp2_apply_ac(accelerator, model: torch.nn.Module):
    """
    Applies the activation checkpointing to the model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to apply the activation checkpointing to

    Returns:
        `torch.nn.Module`: The model with the activation checkpointing applied
    """

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )

    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(accelerator.state.fsdp_plugin, model)

    for layer_name, layer in get_module_children_bottom_up(model, return_fqns=True)[:-1]:
        if len(layer_name.split(".")) > 1:
            parent_name, child_name = layer_name.rsplit(".", 1)
        else:
            parent_name = None
            child_name = layer_name

        parent_module = model.get_submodule(parent_name) if parent_name else model
        if auto_wrap_policy_func(parent_module):
            layer = checkpoint_wrapper(layer, preserve_rng_state=False)
            parent_module.register_module(child_name, layer)

    return model


def fsdp2_prepare_model(
    accelerator,
    model: torch.nn.Module,
    rank_local_parameter_names: Iterable[str] = (),
) -> torch.nn.Module:
    """Prepares the model for FSDP2 in-place. Also returns the model to avoid misuse of the original model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to prepare
        rank_local_parameter_names (`Iterable[str]`, defaults to `()`): Fully qualified names of explicitly registered
            parameters owned outside FSDP2.

    Returns:
        `torch.nn.Module`: Prepared model
    """
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    is_type_fsdp = isinstance(model, FSDPModule) or (
        is_compiled_module(model) and isinstance(model._orig_mod, FSDPModule)
    )
    if is_type_fsdp:
        return model

    fsdp2_plugin = accelerator.state.fsdp_plugin
    if isinstance(rank_local_parameter_names, str):
        raise TypeError("`rank_local_parameter_names` must be an iterable of parameter names, not a string.")
    rank_local_parameter_names = tuple(rank_local_parameter_names)
    if any(not isinstance(name, str) or not name for name in rank_local_parameter_names):
        raise TypeError("`rank_local_parameter_names` must contain only non-empty strings.")
    if len(rank_local_parameter_names) != len(set(rank_local_parameter_names)):
        raise ValueError("`rank_local_parameter_names` must not contain duplicates.")
    named_parameters = fsdp2_canonicalize_names(dict(model.named_parameters()))
    missing_rank_local_names = sorted(set(rank_local_parameter_names).difference(named_parameters))
    del named_parameters
    if missing_rank_local_names:
        raise RuntimeError(f"Registered FSDP2 rank-local parameters are missing: {missing_rank_local_names}")

    fsdp2_plugin.set_auto_wrap_policy(model)

    original_sd = model.state_dict()
    mesh = getattr(accelerator, "torch_device_mesh", None)

    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        # `fully_shard` does not accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
        "mesh": mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)] if mesh is not None else None,
    }

    # `ignored_params` is only supported in torch >= 2.7.0
    model_has_params4bit = False
    incompatible_params4bit = set()
    for name, param in model.named_parameters():
        # this is a temporary fix whereby loading models with bnb params cannot be moved from
        # GPU to a meta device due with FSDP2 because torch operations don't return the original class type
        # bypassing the move to meta will still cause the VRAM spike, but at least it still will load
        if param.__class__.__name__ == "Params4bit":
            model_has_params4bit = True
            # Exclude non-floating frozen Params4bit from FSDP sharding.
            # Default uint8 quant_storage cannot survive fully_shard's DTensor conversion.
            if (not param.requires_grad) and (not param.is_floating_point()) and (not param.is_complex()):
                incompatible_params4bit.add(param)

    if incompatible_params4bit and is_torch_version(">=", "2.7.0"):
        ignored = set(fsdp2_kwargs.get("ignored_params", set()))
        fsdp2_kwargs["ignored_params"] = ignored | incompatible_params4bit
        if accelerator.is_main_process:
            warnings.warn(
                f"Found {len(incompatible_params4bit)} non-floating frozen Params4bit. "
                "Excluding from FSDP2 sharding to prevent quant_state corruption."
                "To enable memory-efficient sharding of 4-bit weights, set"
                "bnb_4bit_quant_storage to a floating dtype (e.g. bf16)."
            )

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # Context: `fully_shard` moves the model to GPU if it was on CPU, however it can also be on `meta` and then it stays there even after `fully_shard`
        # For this reason, we need to move the model to `meta` device, as then sharding happens on `meta` device
        # If we kept the model on CPU (`cpu_ram_efficient_loading` has model be on CPU on all ranks, though non-main ranks only have `torch.empty`), `fully_shard` would move it to GPU
        # Afterwards, when we call `fsdp2_load_full_state_dict`, us creating the state_dict would result into briefly having two copies of model state_dict on the GPU -> VRAM spike

        # We need to keep the original non-persistent buffers, as those MAY not be in the state_dict, resulting in them staying on meta device
        # Also, these buffers aren't getting sharded by default
        # We get the FQNs of all non-persistent buffers, to re-register them after
        non_persistent_buffer_fqns = get_non_persistent_buffers(model, recurse=True, fqns=True)
        original_non_persistent_buffers = copy.deepcopy(
            {k: v for k, v in model.named_buffers() if k in non_persistent_buffer_fqns}
        )
        # We move the model to meta device, as then sharding happens on meta device
        model = model.to(torch.device("meta"))
        # We need to re-tie the weights, not exactly sure why, but if we don't do this, reference to `lm_head/embed_tokens` stay hanging -> more VRAM usage
        # We assume `transformers` models have a `tie_weights` method if they support it
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    if is_torch_version(">=", "2.7.0") and fsdp2_plugin.ignored_modules is not None:
        ignored = set(fsdp2_kwargs.get("ignored_params", set()))
        fsdp2_kwargs["ignored_params"] = ignored | get_parameters_from_modules(
            fsdp2_plugin.ignored_modules, model, accelerator.device
        )
    if rank_local_parameter_names:
        if is_torch_version("<", "2.7.0"):
            raise RuntimeError("FSDP2 rank-local parameters require PyTorch 2.7 or newer.")
        current_named_parameters = fsdp2_canonicalize_names(dict(model.named_parameters()))
        current_rank_local_parameters = {name: current_named_parameters[name] for name in rank_local_parameter_names}
        del current_named_parameters
        ignored = set(fsdp2_kwargs.get("ignored_params", set()))
        fsdp2_kwargs["ignored_params"] = ignored | set(current_rank_local_parameters.values())

    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    if auto_wrap_policy_func is not None:
        # We skip the model itself, as that one is always wrapped
        for module in get_module_children_bottom_up(model)[:-1]:
            if auto_wrap_policy_func(module) and not isinstance(module, FSDPModule):
                fully_shard(module, **fsdp2_kwargs)

    if not isinstance(model, FSDPModule):
        fully_shard(model, **fsdp2_kwargs)

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # If `cpu_ram_efficient_loading` is enabled, only rank 0 loads the weights
        # Other ranks have an empty model on `meta` device, so we need to distribute the weights properly
        # When CPU offloading is enabled, parameters need to stay on CPU after distribution
        from torch.distributed.fsdp import CPUOffloadPolicy

        fsdp2_load_full_state_dict(
            accelerator,
            model,
            original_sd,
            cpu_offload=isinstance(fsdp2_plugin.cpu_offload, CPUOffloadPolicy),
            rank_local_parameter_names=rank_local_parameter_names,
        )

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # We re-register the buffers, as they may not be in the state_dict
        for fqn, buffer_tensor in original_non_persistent_buffers.items():
            buffer_tensor = buffer_tensor.to(accelerator.device)

            if "." in fqn:
                parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
                parent_module = model.get_submodule(parent_fqn)
            else:
                local_buffer_name = fqn
                parent_module = model

            parent_module.register_buffer(local_buffer_name, buffer_tensor, persistent=False)

        # We need to tie the weights again, as call to `load_full_state_dict` breaks the tie
        # Needs to be called both here and above
        # removing this call makes the have slightly different loss
        # removing the call above leads to extra memory usage as explained in the comment above
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # There is no `dtype` attribution for nn.Module
    # Set it to None if it doesn't exist and do the upcast always
    model_dtype = getattr(model, "dtype", None)
    if accelerator.mixed_precision != "no" and (model_dtype is None or model_dtype != torch.float32):
        # We upcast the trainable parameters according to `deepspeed`'s implementation
        # More info about this can be found in `accelerator.py:prepare_model`s FSDP1 section
        upcasted_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                upcasted_params.append(name)
                param = param.to(torch.float32)
        if accelerator.is_main_process and upcasted_params:
            warnings.warn(
                "FSDP upcast of low precision parameters to fp32 (since mixed_precision != 'no') may affect the precision of model checkpoints. "
                f"This effects {len(upcasted_params)} parameters: {upcasted_params}..."
            )
    return model


def fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model: torch.nn.Module) -> Callable[[torch.nn.Module], bool]:
    """Prepares the auto wrap policy based on its type, done to mimic the behaviour of FSDP1 auto wrap policy.

    Args:
        fsdp2_plugin (`FullyShardedDataParallelPlugin`):
            Instance of `FullyShardedDataParallelPlugin` containing the configuration options
        auto_wrap_policy_type (`str`):
            Either `transformer` or `size`
        model (`torch.nn.Module`):
            The model to wrap

    Returns:
        `Callable[[torch.nn.Module], bool]`:
            The auto wrap policy function to be applied to the model
    """
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

    fn = fsdp2_plugin.auto_wrap_policy

    if isinstance(fn, functools.partial):
        fn = fn.func

    if fn is transformer_auto_wrap_policy:
        no_split_modules = getattr(model, "_no_split_modules", None)
        if no_split_modules is None:
            no_split_modules = []
        transformer_cls_names_to_wrap = list(no_split_modules)
        if fsdp2_plugin.transformer_cls_names_to_wrap is not None:
            transformer_cls_names_to_wrap = fsdp2_plugin.transformer_cls_names_to_wrap
        transformer_cls_to_wrap = set()

        for layer_class in transformer_cls_names_to_wrap:
            transformer_cls = get_module_class_from_name(model, layer_class)
            if transformer_cls is None:
                raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
            transformer_cls_to_wrap.add(transformer_cls)

        def policy(module: torch.nn.Module) -> bool:
            if not transformer_cls_to_wrap:
                return False
            return isinstance(module, tuple(transformer_cls_to_wrap))

    elif fn is size_based_auto_wrap_policy:

        def policy(module: torch.nn.Module) -> bool:
            module_num_params = sum(p.numel() for p in module.parameters())
            return module_num_params > fsdp2_plugin.min_num_params
    else:
        return None

    return policy


def get_fsdp2_grad_scaler(**kwargs):
    """
    Returns a `GradScaler` for FSDP2, as the current implementation of `get_grad_scaler` doesn't accept other args. We
    need this as current `get_grad_scaler` accepts only `distributed_type` as arg, which doesn't differentiate between
    FSDP1 and FSDP2
    """
    from torch.amp.grad_scaler import GradScaler

    return GradScaler(**kwargs)


def fsdp2_canonicalize_names(named_params: dict) -> dict:
    """Removes parameter name modifiers in order to map them back to their original names.

    See huggingface/accelerate#3554 for more context.

    Args:
        named_params (`dict`): The named parameters dictionary to canonicalize.

    Returns:
        `dict`: The canonicalized named parameters dictionary
    """
    named_params = {k.replace("._checkpoint_wrapped_module", ""): v for k, v in named_params.items()}
    named_params = {
        k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k: v for k, v in named_params.items()
    }
    named_params = {k.replace("._orig_mod", ""): v for k, v in named_params.items()}
    return named_params


def get_parameters_from_modules(
    modules: Union[Iterable[torch.nn.Module], str], model, device
) -> set[torch.nn.Parameter]:
    """Converts modules to parameters where modules can be a string or list of torch.nn.Module

    Args:
        modules (`Union[Iterable[torch.nn.Module], str]`): List of modules

    Returns:
        `set[torch.nn.Parameter]`: List of parameters
    """
    if modules is None:
        return set()
    parameters = []
    # code taken from accelerate while preparing kwargs for FSDP
    if isinstance(modules, str):
        reg = re.compile(modules)
        mapped_modules = []
        for name, module in model.named_modules():
            if reg.fullmatch(name):
                module.to(device)
                mapped_modules.append(module)
        modules = mapped_modules
    for module in modules:
        parameters.extend(list(module.parameters()))
    return set(parameters)

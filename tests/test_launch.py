# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import argparse
import unittest

from accelerate.utils.launch import _apply_kt_config_to_env, prepare_multi_gpu_env


class TestApplyKTConfigToEnv(unittest.TestCase):
    def test_serializes_activation_policy_for_workers(self):
        args = argparse.Namespace(
            kt_config={
                "enabled": True,
                "kt_activation_policy": {"cpu": "retain", "gpu": "recompute"},
            }
        )

        env = _apply_kt_config_to_env(args, {})

        self.assertEqual(env["ACCELERATE_USE_KT"], "true")
        self.assertEqual(
            env["ACCELERATE_KT_ACTIVATION_POLICY"],
            '{"cpu":"retain","gpu":"recompute"}',
        )

    def test_explicit_worker_env_takes_precedence(self):
        args = argparse.Namespace(
            kt_config={
                "enabled": True,
                "kt_activation_policy": {"cpu": "retain", "gpu": "recompute"},
            }
        )
        existing = '{"cpu":"recompute","gpu":"recompute"}'

        env = _apply_kt_config_to_env(
            args,
            {"ACCELERATE_KT_ACTIVATION_POLICY": existing},
        )

        self.assertEqual(env["ACCELERATE_KT_ACTIVATION_POLICY"], existing)

    def test_disabled_kt_does_not_forward_activation_policy(self):
        args = argparse.Namespace(
            kt_config={
                "enabled": False,
                "kt_activation_policy": {"cpu": "retain", "gpu": "recompute"},
            }
        )

        env = _apply_kt_config_to_env(args, {})

        self.assertEqual(env["ACCELERATE_USE_KT"], "false")
        self.assertNotIn("ACCELERATE_KT_ACTIVATION_POLICY", env)

    def test_forwards_weight_and_training_fields(self):
        args = argparse.Namespace(
            kt_config={
                "enabled": True,
                "kt_expert_weight_format": "int8",
                "kt_weight_lifecycle": "persistent",
                "kt_expert_checkpoint_path": "/weights/experts",
                "kt_non_expert_weight_path": "/weights/non-experts",
                "kt_lora_dropout": 0.1,
                "kt_train_mode": "lora",
            }
        )

        env = _apply_kt_config_to_env(args, {})

        self.assertEqual(env["ACCELERATE_KT_EXPERT_WEIGHT_FORMAT"], "int8")
        self.assertEqual(env["ACCELERATE_KT_WEIGHT_LIFECYCLE"], "persistent")
        self.assertEqual(env["ACCELERATE_KT_EXPERT_CHECKPOINT_PATH"], "/weights/experts")
        self.assertEqual(env["ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH"], "/weights/non-experts")
        self.assertEqual(env["ACCELERATE_KT_LORA_DROPOUT"], "0.1")
        self.assertEqual(env["ACCELERATE_KT_TRAIN_MODE"], "lora")

    def test_forwards_false_boolean(self):
        args = argparse.Namespace(
            kt_config={
                "enabled": True,
                "kt_force_fused_expert_lora": False,
            }
        )

        env = _apply_kt_config_to_env(args, {})

        self.assertEqual(env["ACCELERATE_KT_FORCE_FUSED_EXPERT_LORA"], "false")

    def test_existing_worker_env_takes_precedence_for_runtime_fields(self):
        args = argparse.Namespace(
            kt_config={
                "enabled": True,
                "kt_expert_weight_format": "int8",
                "kt_weight_lifecycle": "ephemeral",
                "kt_non_expert_weight_path": "/config/non-experts",
                "kt_force_fused_expert_lora": False,
            }
        )
        existing = {
            "ACCELERATE_KT_EXPERT_WEIGHT_FORMAT": "bf16",
            "ACCELERATE_KT_WEIGHT_LIFECYCLE": "persistent",
            "ACCELERATE_KT_NON_EXPERT_WEIGHT_PATH": "/env/non-experts",
            "ACCELERATE_KT_FORCE_FUSED_EXPERT_LORA": "true",
        }

        env = _apply_kt_config_to_env(args, existing.copy())

        for key, value in existing.items():
            self.assertEqual(env[key], value)


class TestPrepareMultiGpuEnv(unittest.TestCase):
    def test_auto_port_selection(self):
        args = argparse.Namespace(
            num_processes=1,
            num_machines=1,
            main_process_ip="127.0.0.1",
            main_process_port=0,
            machine_rank=0,
            module=False,
            no_python=False,
            debug=False,
            gpu_ids="all",
            mixed_precision="no",
            dynamo_backend="NO",
            dynamo_mode="default",
            dynamo_use_fullgraph=False,
            dynamo_use_dynamic=False,
            dynamo_use_regional_compilation=False,
            use_fsdp=False,
            fsdp_cpu_ram_efficient_loading=False,
            fsdp_sync_module_states=False,
            fsdp_version=None,
            fsdp_sharding_strategy=None,
            fsdp_reshard_after_forward=False,
            fsdp_offload_params=False,
            fsdp_min_num_params=0,
            fsdp_auto_wrap_policy=None,
            fsdp_transformer_layer_cls_to_wrap=None,
            fsdp_backward_prefetch=None,
            fsdp_state_dict_type=None,
            fsdp_forward_prefetch=False,
            fsdp_use_orig_params=False,
            fsdp_activation_checkpointing=False,
            use_tp=False,
            tp_size=1,
            use_megatron_lm=False,
            megatron_lm_tp_degree=1,
            megatron_lm_pp_degree=1,
            megatron_lm_gradient_clipping=1.0,
            megatron_lm_num_micro_batches=None,
            megatron_lm_sequence_parallelism=None,
            megatron_lm_recompute_activations=None,
            megatron_lm_use_distributed_optimizer=None,
            num_cpu_threads_per_process=1,
            enable_cpu_affinity=False,
            same_network=False,
            use_parallelism_config=False,
        )

        prepare_multi_gpu_env(args)
        self.assertIn("master_port", args.__dict__)
        self.assertNotEqual(args.master_port, "0")
        self.assertTrue(args.master_port.isdigit())

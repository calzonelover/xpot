# source: https://github.com/microsoft/MathOctopus/blob/main/utils/utils.py#L147
import os

import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]

def save_zero_three_model(model_ema, tokenizer, global_rank, save_dir, zero_stage=3):
    zero_stage_3 = zero_stage == 3
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    CONFIG_NAME = "config.json"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, "module") else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, "ds_id"):
                with deepspeed.zero.GatheredParameters(
                    _z3_params_to_fetch([v]), enabled=zero_stage_3
                ):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(save_dir)
        del output_state_dict
import argparse
import copy
import logging
import math
import sys

sys.path.append(".")

import deepspeed
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.integrations import HfDeepSpeedConfig

from xpot.utils.save import save_zero_three_model

from datasets import load_dataset

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def info_rank_0(msg, rank):
    if rank == 0:
        logger.info(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device, dtype=torch.long)
        except Exception as e:
            print(e)
            output[k] = v
    return output


def set_random_seed(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_deepspeed_config(stage, batch_size=4, micro_batch_size=32, max_out_tokens=512):
    return {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": stage,
            "offload_param": {"device": "none"},
            "offload_optimizer": {"device": "none"},
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "memory_efficient_linear": False,
        },
        "bf16": {
            "enabled": True,
            "loss_scale_window": 50,
            "min_loss_scale": 1e-10,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": False,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": 1,
            "release_inference_cache": False,
            "pin_parameters": True,
            "tp_gather_partition_size": 8,
        },
    }


def main(args):
    #################
    # Setup deepspeed
    #################
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    set_random_seed(args.seed)
    torch.distributed.barrier()

    deepspeed_config = get_deepspeed_config(args.zero_stage)
    deepspeed_config["train_micro_batch_size_per_gpu"] = (
        args.per_device_train_batch_size
    )
    deepspeed_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )
    ##############
    # Setup logger
    ##############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    ##########################
    # Load model and tokenizer
    ##########################
    # https://huggingface.co/docs/transformers/deepspeed
    dschf = HfDeepSpeedConfig(
        deepspeed_config
    )  # keep object alive to detect deepspeed for HF
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_seq_length
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16
    )

    # source: https://github.com/microsoft/MathOctopus/blob/main/utils/model/model_utils.py#L14
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make vocab size multiple of 8
    info_rank_0(f"Original vocab size: {len(tokenizer)}", args.global_rank)
    info_rank_0(f"Adjusted vocab size: {model.config.vocab_size}", args.global_rank)

    #################
    # Prepare dataset
    #################
    def prep_text(example):
        output_column = "answer" if args.cot else "teacher_pot"
        if args.train_method.endswith("No-Comment"):
            target_language = "Python" 
        elif args.train_method.startswith("Cross"):
            target_language = "Python with inline comments in English"
        elif args.train_method.startswith("Multi"):
            if args.train_method.endswith("Parallel") or args.train_method.endswith("Cross"):
                target_language = "Python with inline comments in %s"%example["language"] 
            elif args.train_method.endswith("Question"):
                target_language = "Python with inline comments in English"
        else:
            raise AttributeError(f"Recieved invalid train_method expected either Multi or Cross recieved {args.train_method}")

        prompt_no_input = (
            "Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request in {target_language}. Please answer in {target_language}.\n\n"
            "### Instruction:\n{question}\n\n### Response:"
        )

        prompt = prompt_no_input.format_map(example)
        return dict(text=prompt, label=example[output_column])

    def preprocess(example):
        strings = [
            example["text"] + example["label"] + tokenizer.eos_token,
            example["text"],
        ]
        tokenized = [
            tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            for text in strings
        ]
        input_ids, _ = [t.input_ids[0] for t in tokenized]
        input_id_length, instruction_length = [
            t.input_ids.ne(tokenizer.pad_token_id).sum().item() for t in tokenized
        ]
        labels = copy.deepcopy(input_ids)
        labels[:instruction_length] = IGNORE_INDEX
        labels[input_id_length + 1 :] = IGNORE_INDEX
        return dict(input_ids=input_ids, label=labels)

    dataset = (
        load_dataset("json", data_files=args.dataset_name_or_path)
        if args.dataset_name_or_path.endswith("jsonl")
        else load_dataset(args.dataset_name_or_path)
    )
    dataset = dataset.map(prep_text, batched=False)
    dataset = dataset.map(preprocess, batched=False)["train"]
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        sampler=sampler,
        batch_size=args.per_device_train_batch_size,
    )

    batch = next(iter(dataloader))
    for (
        key,
        sample,
    ) in batch.items():
        sample = sample[0].int()
        if key == "input_ids":
            logger.info(f"Sample of training data\nInput:\n{tokenizer.decode(sample)}")
        elif key == "labels":
            label_text = tokenizer.decode(sample[sample.ne(IGNORE_INDEX)])
            logger.info(f"Label:\n{label_text}")

    #######
    # Train
    #######
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), fused=True
    )
    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=deepspeed_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )
    model.gradient_checkpointing_enable()

    for epoch in range(args.num_train_epochs):
        info_rank_0(
            f"Beginning of epoch {epoch+1}/{args.num_train_epochs}, Total micro batches: {len(dataloader)}",
            args.global_rank,
        )
        steps = 0
        model.train()
        for batch in tqdm(dataloader):
            batch = to_device(batch, device)
            output = model(**batch, use_cache=False)
            loss = output.loss
            model.backward(loss)
            model.step()

            steps += 1
            if steps % args.log_steps == 0:
                info_rank_0(
                    f"Epoch: {epoch}, Total Step: {steps}, Loss: {loss}",
                    args.global_rank,
                )

    ############
    # Save model
    ############
    if args.zero_stage == 3:
        save_zero_three_model(model, tokenizer, args.global_rank, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot", action="store_true", default=False, help="whether you are training with CoT data")
    parser.add_argument(
        "--train_method",
        type=str,
        choices=["Cross", "Cross-No-Comment", "Multi-Cross", "Multi-Question", "Multi-Parallel", "Multi-No-Comment"],
        required=True,
        help="Training Method to choose from Cross stands for CrossLingual and Multi stands for MultiLingual")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--zero_stage", type=int, default=3)
    main(parser.parse_args())

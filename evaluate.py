import re
import argparse
import sys
import os
import json
from pprint import pprint

sys.path.append(".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from tqdm import tqdm
from func_timeout import func_timeout 

LANGS = {
    "sw": "Swahili",
    "en": "English",
    "zh": "Chinese",
    "bn": "Bengali",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "ru": "Russian",
    "th": "Thai",
}


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text) 
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

def extract_pot_answer(teacher_pot: str) -> str:
    match = re.search(r"def\s+solver\(\)(.|\n)*?return\s\w+", teacher_pot, re.MULTILINE)
    if match:
        return match.group()
    return None


def get_batch(model, tokenizer):
    @torch.inference_mode()
    def batch(inputs):
        inputs = tokenizer(inputs, padding=True, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=GenerationConfig(
                max_new_tokens=600,
                do_sample=False,
                temperature=0.0
            ),
        ).tolist()
        output_ids = [
            output_id[len(inputs.input_ids[i]) :]
            for i, output_id in enumerate(output_ids)
        ]
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output
    return batch


def main(args):
    set_seed(args.seed)
    ##########################
    # Load model and tokenizer
    ##########################
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    inference = get_batch(model, tokenizer)

    save_dir = f"/ist-project/scads/pumet/xpot/results/{os.path.basename(args.model_name_or_path)}"
    os.makedirs(save_dir, exist_ok=True)

    def get_prompt(train_method, language=None):
        if args.cot:
            target_language = language
        elif train_method.endswith("No-Comment"):
            target_language = "Python" 
        elif train_method.startswith("Cross"):
            target_language = "Python with inline comments in English"
        elif train_method.startswith("Multi"):
            if train_method.endswith("Parallel") or args.train_method.endswith("Cross"):
                target_language = f"Python with inline comments in {language}"
            elif train_method.endswith("Question"):
                target_language = "Python with inline comments in English"
        else:
            raise AttributeError(f"Recieved invalid train_method expected either Multi or Cross recieved {args.train_method}")
        prompt = (
                "Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request in {target_language}. Please answer in {target_language}.\n\n"
                "### Instruction:\n{question}\n\n### Response:"
            )
        return prompt


    results = {}
    for acr, full in LANGS.items():
        dataset = [json.loads(line) for line in open(f"{args.dataset_name_or_path}/mgsm_{acr}.jsonl", 'r', encoding='utf-8')]
        pred_path = os.path.join(save_dir, f"pred_mgsm_{acr}.jsonl")
        prompt = get_prompt(args.train_method, full)
        for i in tqdm(range(0, len(dataset), args.batch_size), desc=f"Generating for {full}"):
            data = dataset[i : i + args.batch_size]
            input_prompts = [prompt.format(question=d["question"]) for d in data]
            outputs = inference(input_prompts)

            for input, output, label in zip(input_prompts, outputs, data):
                with open(pred_path, "a") as f:
                    json.dump(dict(question=input, prediction=output, label=label["label"]), f)
                    f.write("\n")

        predictions = [json.loads(line) for line in open(pred_path, 'r', encoding='utf-8')]
        correct = 0
        for prediction in predictions:
            if args.cot:
                output = extract_last_num(prediction["prediction"])
                if output == float(prediction["label"].replace(",","")):
                    correct += 1
            else:
                pot = extract_pot_answer(prediction["prediction"])
                try:
                    exec(pot)
                    output = float(func_timeout(5, solver))
                    if output == float(prediction["label"].replace(",","")):
                        correct += 1
                except Exception as e:
                    print(e)
                    continue
        results[full] = round(correct / len(predictions), 4)
    pprint(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--cot", action="store_true", default=False, help="whether you are training with CoT data")
    parser.add_argument(
        "--train_method",
        type=str,
        choices=["Cross", "Cross-No-Comment", "Multi-Cross", "Multi-Question", "Multi-Parallel", "Multi-No-Comment"],
        required=True,
        help="Training Method to choose from Cross stands for CrossLingual and Multi stands for MultiLingual")
    parser.add_argument("--self-consistency", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=40)
    main(parser.parse_args())

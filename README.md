# Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments
[![arXiv](https://img.shields.io/badge/arXiv-2502.17956-b31b1b.svg)](https://arxiv.org/abs/2502.17956)

This repository contains code, data, and models for [Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments](https://arxiv.org/abs/2502.17956). Our paper was accepted to ACL Findings 2025.

### Datasets and Models
Our dataset and models are all available at Huggingface

🤗 [airesearch/Cross-and-Multilingual-PoT](https://huggingface.co/collections/airesearch/cross-and-multilingual-pot-6835457ed34b8e5da4d69d6b)
#### CrossLingual
| Model             | Variant                 | mGSM Average |
|-------------------|-------------------------|:------------:|
| **Llama-2-7B**    | [Comment]()             | 30.0         |
|                   | [No Comment]()          | 31.6         |
| **Llama-2-13B**   | [Comment]()             | 36.6         |
|                   | [No Comment]()          | 38.6         |
| **CodeLlama-7B**  | [Comment]()             | 37.4         |
|                   | [No Comment]()          | 39.6         |
| **Llama-3-8B**    | [Comment]()             | 40.6         |
|                   | [No Comment]()          | 53.5         |

#### Multilingual
| Model             | Variant                                                                              | mGSM Average |
|-------------------|--------------------------------------------------------------------------------------|:------------:|
| **Llama-2-7B**    | [PoT Cross Comment](https://huggingface.co/airesearch/Llama-2-7B-Multi-Cross)        |     36.6     |
|                   | [PoT Cross Question](https://huggingface.co/airesearch/Llama-2-7B-Multi-Question)    |     37.7     |
|                   | [PoT Parallel](https://huggingface.co/airesearch/Llama-2-7B-Multi-Parallel)          |     44.6     |
|                   | [PoT No Comment](https://huggingface.co/airesearch/Llama-2-7B-Multi-No-Comment)      |     40.6     |
| **Llama-2-13B**    | [PoT Cross Comment](https://huggingface.co/airesearch/CodeLlama-7B-Multi-Cross)     |     42.2     |
|                   | [PoT Cross Question](https://huggingface.co/airesearch/CodeLlama-7B-Multi-Question)  |     45.1     |
|                   | [PoT Parallel](https://huggingface.co/airesearch/CodeLlama-7B-Multi-Parallel)        |     54.6     |
|                   | [PoT No Comment]https://huggingface.co/airesearch/CodeLlama-7B-Multi-No-Comment)     |     46.4     |
| **CodeLlama-7B**  | [PoT Cross Comment](https://huggingface.co/airesearch/Llama-2-13B-Multi-Cross)       |     41.1     |
|                   | [PoT Cross Question](https://huggingface.co/airesearch/Llama-2-13B-Multi-Question)   |     40.5     |
|                   | [PoT Parallel](https://huggingface.co/airesearch/Llama-2-13B-Multi-Parallel)         |     49.0     |
|                   | [PoT No Comment](https://huggingface.co/airesearch/Llama-2-13B-Multi-No-Comment)     |     45.6     |
| **Llama-3-8B**    | [PoT Cross Comment](https://huggingface.co/airesearch/Llama-3.1-8B-Multi-Cross)      |     58.3     |
|                   | [PoT Cross Question](https://huggingface.co/airesearch/Llama-3.1-8B-Multi-Question)  |     31.6     |
|                   | [PoT Parallel](https://huggingface.co/airesearch/Llama-3.1-8B-Multi-Parallel)        |     62.6     |
|                   | [PoT No Comment](https://huggingface.co/airesearch/Llama-3.1-8B-Multi-No-Comment)    |     56.5     |
| **Qwen2.5-7B**    | [PoT Cross Comment](https://huggingface.co/airesearch/Qwen2.5-7B-Multi-Cross)        |     68.6     |
|                   | [PoT Cross Question](https://huggingface.co/airesearch/Qwen2.5-7B-Multi-Question)    |     62.3     |
|                   | [PoT Parallel](https://huggingface.co/airesearch/Qwen2.5-7B-Multi-Parallel)          |     70.7     |
|                   | [PoT No Comment](https://huggingface.co/airesearch/Qwen2.5-7B-Multi-No-Comment)      |     64.7     |

## Installation
```bash
git clone git@github.com:calzonelover/xpot.git
cd xpot
conda create --name xpot python=3.12
pip install -r requirements.txt
```

## Training
Note: The `METHOD` changes based on the training data (i.e. MGSM8KPoT Parallel expects `METHOD=Multi-Parallel`)
```bash
deepspeed --include localhost:0,1,2,3 train.py \
    --dataset_name_or_path $DATA_PATH \
    --model_name_or_path $MODEL_PATH \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --train_method "$METHOD"
```
## Evaluation
To replicate the experiment results of MGSM in our paper, run:
Note: The `dataset_name_or_path` expects a json file with the keys `question` and `label`
```bash
python evaluate.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_name_or_path $MGSM_PATH \
    --batch_size 16 \
    --method "$METHOD"
```

## Citation
```BibTeX
@misc{payoungkhamdee2025betterunderstandingprogramofthoughtreasoning,
      title={Towards Better Understanding of Program-of-Thought Reasoning in Cross-Lingual and Multilingual Environments}, 
      author={Patomporn Payoungkhamdee and Pume Tuchinda and Jinheon Baek and Samuel Cahyawijaya and Can Udomcharoenchaikit and Potsawee Manakul and Peerat Limkonchotiwat and Ekapol Chuangsuwanich and Sarana Nutanong},
      year={2025},
      eprint={2502.17956},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.17956}, 
}
```

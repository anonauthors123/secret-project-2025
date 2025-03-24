---
library_name: peft
license: other
base_model: LLM4Binary/llm4decompile-1.3b-v2
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: lora_sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lora_sft

This model is a fine-tuned version of [LLM4Binary/llm4decompile-1.3b-v2](https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v2) on the pseudo_code_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1339

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 4
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 6.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0783        | 0.9989 | 580  | 0.5824          |
| 0.0957        | 1.9987 | 1160 | 0.5514          |
| 0.076         | 2.9985 | 1740 | 0.7793          |
| 0.196         | 4.0    | 2120 | 0.2314          |
| 0.117         | 5.0    | 2650 | 0.1371          |
| 0.0543        | 6.0    | 3180 | 0.1339          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.20.3
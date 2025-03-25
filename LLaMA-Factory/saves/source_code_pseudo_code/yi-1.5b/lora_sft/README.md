---
library_name: peft
license: other
base_model: 01-ai/Yi-Coder-1.5B-Chat
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

This model is a fine-tuned version of [01-ai/Yi-Coder-1.5B-Chat](https://huggingface.co/01-ai/Yi-Coder-1.5B-Chat) on the pseudo_code_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1073

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
| 0.0579        | 0.9989 | 580  | 0.3372          |
| 0.0447        | 1.9987 | 1160 | 0.4925          |
| 0.0434        | 2.9985 | 1740 | 0.6921          |
| 0.1402        | 4.0    | 2120 | 0.1643          |
| 0.0794        | 5.0    | 2650 | 0.1073          |
| 0.0654        | 6.0    | 3180 | 0.1084          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.46.1
- Pytorch 2.5.1+cu124
- Datasets 3.1.0
- Tokenizers 0.20.3
import os
import subprocess
import time

ENVS = {
    "CUDA_VISIBLE_DEVICES": "0,1",
    # "HF_HUB_OFFLINE": "1",
}

MODELS = [
    "Qwen2.5-Coder-0.5B-Instruct",
    
    "deepseek-coder-1.3b-instruct",
    "llm4decompile-1.3b-v2",
    "OpenCoder-1.5B-Instruct",
    "Qwen2.5-Coder-1.5B-Instruct",
    "Yi-Coder-1.5B-Chat",
    
    "Qwen2.5-Coder-3B-Instruct",
    
    "llm4decompile-6.7b-v2",
    "magicoder-s-ds-6.7b",
    "CodeLlama-7b-Instruct-hf",
    "Qwen2.5-Coder-7B-Instruct",
    "aixcoder-7b-base",
    "deepseek-coder-7b-instruct-v1.5",
    "llm-compiler-7b",
    "llm-compiler-7b-ftd",
    "starcoder2-7b",
    "OpenCoder-8B-Instruct",
    "Yi-Coder-9B-Chat",
    "llm4decompile-9b-v2",
]


CONFIGS = [
    # "src.yaml",
    # "src_pred.yaml",
    # "src_pseudo_stage2.yaml",
    # "src_pseudo_stage2_pred.yaml",
    
    "src_pseudo_mixed.yaml",
    "src_pseudo_mixed_pred.yaml",
    
    "pseudo.yaml",
    "pseudo_pred.yaml",
    "asm.yaml",
    "asm_pred.yaml",
]


for model in MODELS:
    for config in CONFIGS:
        cmd = f"llamafactory-cli train ./configs/{model}/{config}"
        print(f"Running: {cmd}")
        os.system(f"{' '.join([f'{k}={v}' for k, v in ENVS.items()])} {cmd}")
        time.sleep(10)
    
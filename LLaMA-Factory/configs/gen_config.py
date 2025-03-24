import os
import re


def get_simple_name(text):
    s = [line for line in text.split("\n") if line.startswith("output_dir: ")][0]
    return s.split("/")[-2]


def gen_src_only():
    for model in os.listdir("./"):
        if not os.path.isdir(model):
            continue
        
        with open(f"./{model}/pseudo.yaml", "r") as f:
            pseudo = f.read()
        
        src_only = pseudo.replace("pseudo_code_train", "source_code_train")
        # src_only = src_only.replace("pseudo_code_dev", "source_code_dev")
        src_only = src_only.replace("output_dir: saves/pseudo_code/", "output_dir: saves/source_code_only/")
        
        with open(f"./{model}/src.yaml", "w") as f:
            f.write(src_only)
        
        
        with open(f"./{model}/pseudo_pred.yaml", "r") as f:
            pseudo_pred = f.read()
        
        src_pred = pseudo_pred.replace("adapter_name_or_path: saves/pseudo_code/", "adapter_name_or_path: saves/source_code_only/")
        src_pred = src_pred.replace("output_dir: saves/pseudo_code/", "output_dir: saves/source_code_only/")
        
        with open(f"./{model}/src_pred.yaml", "w") as f:
            f.write(src_pred)


def gen_src_pseudo_mixed():
    for model in os.listdir("./"):
        if not os.path.isdir(model):
            continue
        
        with open(f"./{model}/pseudo.yaml", "r") as f:
            pseudo = f.read()
        
        mixed = pseudo.replace("pseudo_code_train", "pseudo_code_train,source_code_train")
        # mixed = mixed.replace("pseudo_code_dev", "pseudo_code_dev,source_code_dev")
        mixed = mixed.replace("output_dir: saves/pseudo_code/", "output_dir: saves/source_code_mix_pseudo_code/")
        
        with open(f"./{model}/src_pseudo_mixed.yaml", "w") as f:
            f.write(mixed)
        
        
        with open(f"./{model}/pseudo_pred.yaml", "r") as f:
            pseudo_pred = f.read()
        
        mixed_pred = pseudo_pred.replace("adapter_name_or_path: saves/pseudo_code/", "adapter_name_or_path: saves/source_code_mix_pseudo_code/")
        mixed_pred = mixed_pred.replace("output_dir: saves/pseudo_code/", "output_dir: saves/source_code_mix_pseudo_code/")
        
        with open(f"./{model}/src_pseudo_mixed_pred.yaml", "w") as f:
            f.write(mixed_pred)


def gen_src_pseudo_stage2():
    for model in os.listdir("./"):
        if not os.path.isdir(model):
            continue
        
        with open(f"./{model}/pseudo.yaml", "r") as f:
            pseudo = f.read()
        
        simple_name = get_simple_name(pseudo)
        stage = pseudo.replace("output_dir: saves/pseudo_code/", "output_dir: saves/source_code_pseudo_code/")
        stage = stage.replace("### method\n", f"### method\nresume_from_checkpoint: saves/source_code_only/{simple_name}/lora_sft/checkpoint-1740\n")
        stage = stage.replace("num_train_epochs: 3.0", "num_train_epochs: 6.0")
        
        with open(f"./{model}/src_pseudo_stage2.yaml", "w") as f:
            f.write(stage)
        
        
        with open(f"./{model}/pseudo_pred.yaml", "r") as f:
            pseudo_pred = f.read()
        
        stage_pred = pseudo_pred.replace("adapter_name_or_path: saves/pseudo_code/", "adapter_name_or_path: saves/source_code_pseudo_code/")
        stage_pred = stage_pred.replace("output_dir: saves/pseudo_code/", "output_dir: saves/source_code_pseudo_code/")
        
        with open(f"./{model}/src_pseudo_stage2_pred.yaml", "w") as f:
            f.write(stage_pred)


if __name__ == "__main__":
    gen_src_only()
    gen_src_pseudo_mixed()
    gen_src_pseudo_stage2()
    
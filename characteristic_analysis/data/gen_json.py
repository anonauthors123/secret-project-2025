import os
from os.path import join as pjoin
import json
from tqdm import tqdm

ASM_ROOT = "../../data/raw_data/CFGs"
PSEUDO_ROOT = "../../data/raw_data/Pseudos"
SRC_ROOT = "../../data/raw_data/Srcs"

ASM_JSON_FILE = "./asm.json"
PSEUDO_JSON_FILE = "./pseudo.json"
SRC_JSON_FILE = "./src.json"


def gen_asm_json():
    cfg_json = []
    for repo in tqdm(os.listdir(ASM_ROOT), total=len(os.listdir(ASM_ROOT))):
        for optim in os.listdir(pjoin(ASM_ROOT, repo)):
            for cfg in os.listdir(pjoin(ASM_ROOT, repo, optim)):
                with open(pjoin(ASM_ROOT, repo, optim, cfg), "r") as f:
                    data = json.load(f)
                code = ""
                for node in data["nodes"]:
                    code += "\n".join(data["nodes"][node]) + "\n"
                cfg_json.append({
                    "repo": repo,
                    "optimize_level": optim,
                    "from": cfg,
                    "code": code
                })
                
    with open(ASM_JSON_FILE, "w") as f:
        json.dump(cfg_json, f, indent=4)
        

def gen_pseudo_json():
    cfg_json = []
    for repo in tqdm(os.listdir(PSEUDO_ROOT), total=len(os.listdir(PSEUDO_ROOT))):
        for optim in os.listdir(pjoin(PSEUDO_ROOT, repo)):
            for cfg in os.listdir(pjoin(PSEUDO_ROOT, repo, optim)):
                with open(pjoin(PSEUDO_ROOT, repo, optim, cfg), "r") as f:
                    data = json.load(f)
                code = "\n".join(data["pseudo_code"])
                cfg_json.append({
                    "optimize_level": optim,
                    "from": cfg,
                    "code": code
                })
    
    with open(PSEUDO_JSON_FILE, "w") as f:
        json.dump(cfg_json, f, indent=4)                



def gen_src_json():
    src_json = []
    for vul_fix in tqdm(os.listdir(SRC_ROOT), total=len(os.listdir(SRC_ROOT))):
        for src in os.listdir(pjoin(SRC_ROOT, vul_fix)):
            with open(pjoin(SRC_ROOT, vul_fix, src), "r") as f:
                data = json.load(f)
            src_json.append({
                "from": f"{vul_fix}_{src}",
                "code": data["function"]
            })

    with open(SRC_JSON_FILE, "w") as f:
        json.dump(src_json, f, indent=4)


def main():
    gen_asm_json()
    gen_pseudo_json()
    gen_src_json()


if __name__ == "__main__":
    main()
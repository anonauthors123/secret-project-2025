import json, jsonlines, os

indexs = []

with jsonlines.open(r"../LLaMA-Factory/saves/source_code_mix_pseudo_code/llmcompiler-7b-ftd/lora_sft_pred/generated_predictions.jsonl") as f:
    for index, line in enumerate(f):
        # print(line)
        if 'no' in line["predict"] and line["label"] == "yes":
            indexs.append(index)

with open(r"3_LLMCompiler-7B-ftd_error_index", 'w') as f:
    json.dump(indexs, f, indent=4)

# with open(r"test_infos.json", 'r') as f:
#     infos = json.load(f)

# error_pred_items = []
# for idx, item in enumerate(infos):
#     if idx in indexs:
#         error_pred_items.append(item)

# with open(r"LLMCompiler-7B _error_predictions.json", 'w') as f:
#     json.dump(error_pred_items, f, indent=4)

                                  
        
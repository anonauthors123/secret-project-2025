import os
import json
import time
import jieba
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
from sklearn.manifold import TSNE


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


ASM_DATASET_PATH = "./data/asm.json"
PSEUDO_DATASET_PATH = "./data/pseudo.json"
SRC_DATASET_PATH = "./data/src.json"

RESULT_DIR = "./results"

PROB_MODEL_NAME = "Salesforce/codet5-large"
EMB_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
CUTOFF_LENGTH = 4096
BATCH_SIZE = 128

NATRUNALNESS_SAMPLE_INTERVAL = 10
NATRUNALNESS_CUTOFF_LENGTH = 512


device = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


prob_tokenizer = None # lazy load
def count_single_token_length(text):
    global prob_tokenizer
    if prob_tokenizer is None:
        print(f"Loading tokenizer {EMB_MODEL_NAME}")
        prob_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    input_ids = prob_tokenizer(text, return_tensors=None, padding=False, add_special_tokens=False)["input_ids"]
    cutoff_input_ids = prob_tokenizer(text, return_tensors=None, padding=False, add_special_tokens=False, max_length=CUTOFF_LENGTH, truncation=True)["input_ids"]
    return len(input_ids), len(cutoff_input_ids)


emb_model = None
def get_single_emb(text):
    global emb_model, prob_tokenizer
    if prob_tokenizer is None:
        print(f"Loading tokenizer {EMB_MODEL_NAME}")
        prob_tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    if emb_model is None:
        print(f"Loading model {EMB_MODEL_NAME}")
        emb_model = AutoModel.from_pretrained(EMB_MODEL_NAME, trust_remote_code=True).to(device)
    model_input = prob_tokenizer(text, return_tensors="pt", padding="max_length", add_special_tokens=True, max_length=CUTOFF_LENGTH, truncation=True).to(device)
    with torch.no_grad():
        model_output = emb_model(**model_input)
    emb = mean_pooling(model_output, model_input["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1).cpu().numpy()
    return emb[0]


prob_tokenizer = None
prob_model = None
def get_single_probs(text, mode = "mlm"):
    global prob_tokenizer, prob_model
    if prob_tokenizer is None:
        print(f"Loading tokenizer {PROB_MODEL_NAME}")
        prob_tokenizer = AutoTokenizer.from_pretrained(PROB_MODEL_NAME)
    if prob_model is None:
        print(f"Loading model {PROB_MODEL_NAME}")
        prob_model = T5ForConditionalGeneration.from_pretrained(PROB_MODEL_NAME).to(device)
    
    
    inputs = prob_tokenizer(text, return_tensors="pt", add_special_tokens=True, max_length=NATRUNALNESS_CUTOFF_LENGTH, truncation=True)
    probs = np.array([])
    with torch.no_grad():
        for i in range(1, len(inputs.input_ids[0]) - 1, BATCH_SIZE):
            end_idx = min(i + BATCH_SIZE, len(inputs.input_ids[0]) - 1)
            original_ids = inputs.input_ids[0][i:end_idx]
            real_inputs = prob_tokenizer([text] * (end_idx - i), return_tensors="pt", add_special_tokens=True, max_length=NATRUNALNESS_CUTOFF_LENGTH, truncation=True)
            for j in range(end_idx - i):
                real_inputs.input_ids[j][i+j] = 32099
                if mode == "clm":
                    real_inputs.input_ids[j][i+j+1] = prob_tokenizer.eos_token_id
                    real_inputs.input_ids[j][i+j+2:] = prob_tokenizer.pad_token_id
                    real_inputs.attention_mask[j][i+j+2:] = 0
            real_inputs.to(device)
            time1 = time.time()
            outputs = prob_model.generate(**real_inputs, max_new_tokens=2, return_dict_in_generate=True, output_logits=True)
            time2 = time.time()
            # print(f"Time taken: {time2 - time1}")
            this_probs = F.softmax(outputs.logits[1], dim=-1)
            this_probs = [this_probs[j][original_ids[j]].item() for j in range(end_idx - i)]
            probs = np.concatenate((probs, this_probs))

    cross_entropy = -np.sum(np.log(probs)) / (len(inputs.input_ids[0]) - 2)
    return cross_entropy


def get_single_tf(corpus):
    #   统计每个词出现的频率
    word_freq_dict = dict()
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 1
        word_freq_dict[word] += 1
    # 将这个词典中的词，按照出现次数排序，出现次数越高，排序越靠前
    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x:x[1], reverse=True)
    # 计算TF概率
    word_tf = dict()
    # 信息熵
    shannoEnt = 0.0
    # 按照频率，从高到低，开始遍历，并未每个词构造一个id
    for word, freq in word_freq_dict:
        # 计算p(xi)
        prob = freq / len(corpus)
        word_tf[word] = prob
        shannoEnt -= prob*np.log2(prob)
    return word_tf, shannoEnt


def analyze_length(dataset, dataset_name):
    token_length_result_dir = os.path.join(RESULT_DIR, "token_length")
    os.makedirs(token_length_result_dir, exist_ok=True)

    if os.path.exists(os.path.join(token_length_result_dir, f"{dataset_name}_token_length.json")):
        with open(os.path.join(token_length_result_dir, f"{dataset_name}_token_length.json"), "r") as f:
            data = json.load(f)
        token_length_list = data["token_length"]
        cutoff_token_length_list = data["cutoff_token_length"]
    else:
        token_length_list = []
        cutoff_token_length_list = []
        for data in tqdm(dataset, total=len(dataset), desc="Counting token length"):
            text = data["code"]
            token_length, cutoff_token_length = count_single_token_length(text)
            token_length_list.append(token_length)
            cutoff_token_length_list.append(cutoff_token_length)
        with open(os.path.join(token_length_result_dir, f"{dataset_name}_token_length.json"), "w") as f:
            json.dump({"token_length": token_length_list, "cutoff_token_length": cutoff_token_length_list}, f)
            
    
    def draw_token_plot(token_length_list, name):
        fig, ax1 = plt.subplots()

        # Plot histogram on ax1
        ax1.hist(token_length_list, bins=50, alpha=0.6, color='g', label='Token Length Distribution')
        ax1.set_xlabel('Token Length')
        ax1.set_ylabel('Frequency', color='g')
        ax1.tick_params(axis='y', labelcolor='g')

        # Create a second y-axis for the cumulative distribution
        ax2 = ax1.twinx()
        cumulative = np.cumsum(np.histogram(token_length_list, bins=50)[0])
        cumulative = cumulative / cumulative[-1]  # Normalize to get percentages
        ax2.plot(np.linspace(min(token_length_list), max(token_length_list), 50), cumulative, color='b', label='Cumulative Distribution')
        ax2.set_ylabel('Cumulative Percentage', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # Calculate statistics
        median = np.median(token_length_list)
        mean = np.mean(token_length_list)
        percentage_4096 = np.sum(np.array(token_length_list) <= 4096) / len(token_length_list) * 100
        # length_90 = np.percentile(token_length_list, 90)

        # Plot statistics
        plt.axvline(median, color='r', linestyle='dashed', linewidth=1, label=f'Median: {median}')
        plt.axvline(mean, color='y', linestyle='dashed', linewidth=1, label=f'Mean: {mean}')
        if max(token_length_list) > 4096:
            plt.axvline(4096, color='purple', linestyle='dashed', linewidth=1, label=f'4096: {percentage_4096:.2f}%')
        # plt.axvline(length_90, color='orange', linestyle='dashed', linewidth=1, label=f'99%: {length_90:.2f}')

        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.title(f'Token Length Analysis for {dataset_name}')
        plt.legend()
        plt.grid(True)

        # Save plot
        plt.savefig(os.path.join(token_length_result_dir, name))
        plt.close()
    
    draw_token_plot(token_length_list, f"{dataset_name}_token_length.png")
    draw_token_plot(cutoff_token_length_list, f"{dataset_name}_cutoff_token_length.png")
    
    optim_length_list = {'O0': [], 'O1': [], 'O2': [], 'O3': [], 'Os': []}
    for i, data in enumerate(dataset):
        if "optimize_level" in data:
            optim_length_list[data["optimize_level"]].append(cutoff_token_length_list[i])
    
    statistics = {
        "all": {
            "max_token_length": max(token_length_list),
            "min_token_length": min(token_length_list),
            "mean_token_length": np.mean(token_length_list),
            "median_token_length": np.median(token_length_list),
            "mean_cutoff_token_length": np.mean(cutoff_token_length_list),
            "4096_percentage": np.sum(np.array(token_length_list) <= 4096) / len(token_length_list) * 100,
        }
    }
    
    for opt_level in optim_length_list:
        if len(optim_length_list[opt_level]) == 0:
            continue
        statistics[opt_level] = {
            "max_token_length": max(optim_length_list[opt_level]),
            "min_token_length": min(optim_length_list[opt_level]),
            "mean_token_length": np.mean(optim_length_list[opt_level]),
            "median_token_length": np.median(optim_length_list[opt_level]),
            "4096_percentage": np.sum(np.array(optim_length_list[opt_level]) <= 4096) / len(optim_length_list[opt_level]) * 100,
        }
    
    with open(os.path.join(token_length_result_dir, f"{dataset_name}_token_length_statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4)
    
    return dataset


def analyze_emb_space(dataset, dataset_name):
    emb_result_dir = os.path.join(RESULT_DIR, "emb_space")
    os.makedirs(emb_result_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(emb_result_dir, f"{dataset_name}_emb_space.npy")):
        emb_list = np.load(os.path.join(emb_result_dir, f"{dataset_name}_emb_space.npy"))
    else:
        emb_list = []
        for data in tqdm(dataset, total=len(dataset), desc="Getting embeddings"):
            emb = get_single_emb(data["code"])
            emb_list.append(emb)
        emb_list = np.array(emb_list)
        np.save(os.path.join(emb_result_dir, f"{dataset_name}_emb_space.npy"), emb_list)
    


def analyze_naturalness(dataset, dataset_name):
    nat_result_dir = os.path.join(RESULT_DIR, "naturalness")
    os.makedirs(nat_result_dir, exist_ok=True)
    
    dataset = dataset[::NATRUNALNESS_SAMPLE_INTERVAL]
    
    if os.path.exists(os.path.join(nat_result_dir, f"{dataset_name}_naturalness.npy")):
        naturalness_list = np.load(os.path.join(nat_result_dir, f"{dataset_name}_naturalness.npy"))
    else:
        naturalness_list = []
        for data in tqdm(dataset, total=len(dataset), desc="Getting naturalness"):
            naturalness = get_single_probs(data["code"], "mlm")
            naturalness_list.append(naturalness)
        naturalness_list = np.array(naturalness_list)
        np.save(os.path.join(nat_result_dir, f"{dataset_name}_naturalness.npy"), naturalness_list)
    
    # print(naturalness_list)
    optim_naturalness_list = {'O0': [], 'O1': [], 'O2': [], 'O3': [], 'Os': []}
    for i, data in enumerate(dataset):
        if "optimize_level" in data:
            optim_naturalness_list[data["optimize_level"]].append(naturalness_list[i])
    
    statistics = {
        "all": {
            "max": np.max(naturalness_list),
            "min": np.min(naturalness_list),
            "median": np.median(naturalness_list),
            "mean": np.average(naturalness_list),
            "std": np.std(naturalness_list),
            "upper_quartile": np.percentile(naturalness_list, 75),
            "lower_quartile": np.percentile(naturalness_list, 25),
        }
    }
    
    for opt_level in optim_naturalness_list:
        if len(optim_naturalness_list[opt_level]) == 0:
            continue
        statistics[opt_level] = {
            "max": np.max(optim_naturalness_list[opt_level]),
            "min": np.min(optim_naturalness_list[opt_level]),
            "median": np.median(optim_naturalness_list[opt_level]),
            "mean": np.average(optim_naturalness_list[opt_level]),
            "std": np.std(optim_naturalness_list[opt_level]),
            "upper_quartile": np.percentile(optim_naturalness_list[opt_level], 75),
            "lower_quartile": np.percentile(optim_naturalness_list[opt_level], 25),
        }
    
    with open(os.path.join(nat_result_dir, f"{dataset_name}_statistics.json"), "w") as f:
        json.dump(statistics, f, indent=4)



def analyze_info_entropy(dataset, dataset_name):
    entropy_result_dir = os.path.join(RESULT_DIR, "info_entropy")
    os.makedirs(entropy_result_dir, exist_ok=True)
    
    global prob_tokenizer
    if prob_tokenizer is None:
        print(f"Loading tokenizer {PROB_MODEL_NAME}")
        prob_tokenizer = AutoTokenizer.from_pretrained(PROB_MODEL_NAME)
    
    if os.path.exists(os.path.join(entropy_result_dir, f"{dataset_name}_info_entropy.json")):
        return
    
    optim_level_list = ['all', 'O0', 'O1', 'O2', 'O3', 'Os']
    entropy = {}
    for optim_level in optim_level_list:
        corpus = []
        for data in tqdm(dataset, total=len(dataset), desc="Getting corpus"):
            if "optimize_level" not in data:
                data["optimize_level"] = 'all'
            if optim_level != 'all' and data["optimize_level"] != optim_level:
                continue
            if not "corpus" in data:
                text = data["code"]
                # text = text.replace("\n", '').replace(" ", '')
                # data["corpus"] = jieba.lcut_for_search(text)
                ids = prob_tokenizer(text, return_tensors=None, padding=False, add_special_tokens=False)["input_ids"]
                data["corpus"] = prob_tokenizer.convert_ids_to_tokens(ids)
            corpus.extend(data["corpus"])
        if len(corpus) == 0:
            continue
        word_tf, shannoEnt = get_single_tf(corpus)
        entropy[optim_level] = {
            # "word_tf": word_tf,
            "shannoEnt": shannoEnt
        }
    
    with open(os.path.join(entropy_result_dir, f"{dataset_name}_info_entropy.json"), "w") as f:
        json.dump(entropy, f, indent=4)   



def analyze_data(dataset_path):
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    
    analyze_length(dataset, dataset_name)
    analyze_emb_space(dataset, dataset_name)
    analyze_naturalness(dataset, dataset_name)
    analyze_info_entropy(dataset, dataset_name)
    


def plot_tsne(dataset_emb_file_map):
    emb_result_dir = os.path.join(RESULT_DIR, "emb_space")
    os.makedirs(emb_result_dir, exist_ok=True)
    
    dataset_emb_map = {}
    dataset_tsne_emb_map = {}
    dataset_size_map = {}
    for dataset in dataset_emb_file_map:
        dataset_emb_map[dataset] = np.load(dataset_emb_file_map[dataset])
        dataset_size_map[dataset] = dataset_emb_map[dataset].shape[0]
    
    if os.path.exists(os.path.join(emb_result_dir, "tsne.npy")):
        tsne_result = np.load(os.path.join(emb_result_dir, "tsne.npy"))
    else:
        emb_stack = np.vstack(list(dataset_emb_map.values()))
        tsne = TSNE(n_components=2, init='pca', random_state=0, verbose=1)
        tsne_result = tsne.fit_transform(emb_stack)
        x_min, x_max = np.min(tsne_result, 0), np.max(tsne_result, 0)
        tsne_result = (tsne_result - x_min) / (x_max - x_min)
        np.save(os.path.join(emb_result_dir, "tsne.npy"), tsne_result)
    
    start_idx = 0
    for dataset in dataset_size_map:
        end_idx = start_idx + dataset_size_map[dataset]
        dataset_tsne_emb_map[dataset] = tsne_result[start_idx:end_idx]
        start_idx = end_idx
        
    tsne_centroids = {}
    for dataset, tsne_data in dataset_tsne_emb_map.items():
        tsne_centroids[dataset] = np.mean(tsne_data, axis=0)
    
    plt.figure(figsize=(10, 8))
        
    colors = ['#e7809e', '#566cfb', '#51d092']
    edge_colors = ['red', 'navy', 'black']
    labels = list(dataset_emb_file_map.keys())
    for i, (dataset, tsne_data) in enumerate(dataset_tsne_emb_map.items()):
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=colors[i], label=labels[i], alpha=0.5, s=25)
    for i, (dataset, centroid) in enumerate(tsne_centroids.items()):
        plt.scatter(centroid[0], centroid[1], c=colors[i], marker='*', s=500, label=f'{dataset} cent.', edgecolors=edge_colors[i], linewidth=2)
    
    
    asm_pseudo_dist = np.linalg.norm(tsne_centroids["asm"] - tsne_centroids["pseudo"])
    src_pseudo_dist = np.linalg.norm(tsne_centroids["src"] - tsne_centroids["pseudo"])
    asm_src_dist = np.linalg.norm(tsne_centroids["asm"] - tsne_centroids["src"])
    
    print(f"assembly-pseudo centroid distance: {asm_pseudo_dist}")
    print(f"source-pseudo centroid distance: {src_pseudo_dist}")
    print(f"assembly-source centroid distance: {asm_src_dist}")
    
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # plt.title('t-SNE Embedding Space', fontsize=26)
    
    # plt.text(0.5, -0.2, f'pseudo-assembly centroid distance: {asm_pseudo_dist:.2f}\npseudo-source centroid distance: {src_pseudo_dist:.2f}\nassembly-source centroid distance: {asm_src_dist:.2f}',
    #          fontsize=18, ha='center', va='bottom', transform=plt.gca().transAxes)
    plt.legend(fontsize=22, ncol=2, markerscale=1.5, loc='lower left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(emb_result_dir, "tsne_scatter_plot.png"))
    plt.savefig(os.path.join(emb_result_dir, "tsne_scatter_plot.pdf"))
    plt.close()



def plot_naturalness(dataset_naturalness_file_map):
    nat_result_dir = os.path.join(RESULT_DIR, "naturalness")
    os.makedirs(nat_result_dir, exist_ok=True)
    
    dataset_naturalness_map = {}
    for dataset in dataset_naturalness_file_map:
        dataset_naturalness_map[dataset] = np.load(dataset_naturalness_file_map[dataset])
        mean = np.mean(dataset_naturalness_map[dataset])
        std = np.std(dataset_naturalness_map[dataset])
        filtered_naturalness = dataset_naturalness_map[dataset][np.abs(dataset_naturalness_map[dataset] - mean) <= 4 * std]
        print(f"Filtered {dataset} samples: {len(dataset_naturalness_map[dataset]) - len(filtered_naturalness)}")
        dataset_naturalness_map[dataset] = filtered_naturalness
    
    plt.figure(figsize=(10, 4))
    data = [dataset_naturalness_map[dataset] for dataset in dataset_naturalness_file_map]
    labels = list(dataset_naturalness_file_map.keys())
    
    bplt = plt.boxplot(data, vert=False, tick_labels=labels, patch_artist=True, medianprops=dict(color='black'))
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    
    colors = ['#e7809e', '#566cfb', '#51d092']
    for dataset, color in zip(bplt['boxes'], colors):
        dataset.set_facecolor(color)
    
    # plt.xlabel('CE', fontsize=20)
    # plt.ylabel('Dataset', fontsize=20)
    # plt.title('Naturalness Analysis')
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(nat_result_dir, "naturalness_box_plot.png"))
    plt.savefig(os.path.join(nat_result_dir, "naturalness_box_plot.pdf"))
    plt.close()


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    analyze_data(ASM_DATASET_PATH)
    analyze_data(PSEUDO_DATASET_PATH)
    analyze_data(SRC_DATASET_PATH)
    
    plot_tsne({
        "asm": os.path.join(RESULT_DIR, "emb_space", "asm_emb_space.npy"),
        "pseudo": os.path.join(RESULT_DIR, "emb_space", "pseudo_emb_space.npy"),
        "src": os.path.join(RESULT_DIR, "emb_space", "src_emb_space.npy"),
    })
    
    plot_naturalness({
        "asembly code": os.path.join(RESULT_DIR, "naturalness", "asm_naturalness.npy"),
        "pseudo code": os.path.join(RESULT_DIR, "naturalness", "pseudo_naturalness.npy"),
        "source code": os.path.join(RESULT_DIR, "naturalness", "src_naturalness.npy"),
    })


if __name__ == "__main__":
    main()

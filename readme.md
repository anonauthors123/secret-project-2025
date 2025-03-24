# Empirical Study of Code LLMs for Binary Security Patch Detection

This is the repository of the ICSE paper *Empirical Study of Code LLMs for Binary Security Patch Detection*. The README document are organized as follows:

## Environment Setup

1. Execute the following command to change to LLaMA-Factory directory and install Python packages for LLaMA-Factory

   ```sh
   cd LLaMA-Factory
   pip install -e ".[torch,metrics]"
   ```

2. In the **root directory**, execute this command to install the rest of the Python Package

   ```sh
   pip install -r requirement.txt
   ```

## Dataset Preparation

We provide raw data `raw_data` and preprocessed (split) datasets in alpaca format `alpaca`. 

Change to `data` directory, and unzip the two dataset.

```sh
cd data
tar -xvjf raw_data.tar.bz2
tar -xvjf alpaca_data.tar.bz2
```

> [!NOTE]
>
> Only part of the dataset is available now. We will release the full size datasets upon acceptance of this paper.

## Finetuning and Inference with LLMs

### Finetuning and Inference with Open-source LLMs

1. Change to LLaMA-Factory directory

   ```sh
   cd LLaMA-Factory
   ```

2. Execute the `@onekey.py` Python script to start one-key finetuning and evaluating

   ```sh
   python @onekey.py
   ```

3. Model weights, prediction results, checkpoints and other result files will be stored in `LLaMA-Factory/saves` directory.

In the `@onekey.py` script, you may select which **models** to experiment on by modifying the `MODELS` list, and which datasets experiment on by modifying the `CONFIGS` list.

> [!NOTE]
>
> - **Config files** are stored in `LLaMA-Factory/configs` directory. 
>   - Files with `_pred.yaml` suffix are used for prediction on the test set.
>   - Please note that we use **2 GPUs for finetuning**. If you need to **reproduce the source-code-then-pseudo-code experiment** with **a different number of GPUs**, the number of training steps may change and you may need to modify the config files to load the correct checkpoint in the second stage training. 
>     Please kindly refer to `LLaMA-Factory/configs/*/src_pseudo_stage2.yaml` and change the value of `resume_from_checkpoint` from `.../checkpoint-1740` to the correct training step.
> - LLaMA-Factory accepts Alpaca format dataset. Datasets in Alpaca format should be placed in `LLaMA-Factory/data` directory.
> - Due to the large size of model weights, we have to exclude them from the repository. 
>   In despite of this, we provide **the training logs and the prediction results** in our repository.

### Inference with Online LLMs

1. Change to `LLM-online` directory

   ```sh
   cd LLm-online
   ```

   Modify the `@onekey.sh` script to set your correct api key and api endpoint url.

   ```sh
   # Usage: bash @onekey.sh
   # -u [api_url]
   # -k [api_key]
   # -m [model_name]
   # -d [data_type]
   # -t [max_token]
   
   HF_HUB_OFFLINE=1
   
   python llm_online.py -u https://api.chatanywhere.tech/v1/ -k [api_key] -m gpt-3.5-turbo -d pseudo_code -t 16384
   python llm_online.py -u https://api.chatanywhere.tech/v1/ -k [api_key] -m gpt-3.5-turbo -d assembly_code -t 16384
   
   python llm_online.py -u https://api.deepseek.com/v1 -k [deepseek_api_key] -m deepseek-reasoner -d pseudo_code -t 65536
   python llm_online.py -u https://api.deepseek.com/v1 -k [deepseek_api_key] -m deepseek-reasoner -d assembly_code -t 65536
   ```

2. Execute the `@onekey.sh` shell script. The prediction result will be stored in `LLM-online/saves` directory.

   ```sh
   bash @onekey.sh
   ```

> [!NOTE]
>
> The LLMs may fail to inference on certain samples. The progress will be stored and you can execute the inference script multiple times to retry on these samples.

## Evaluation

1. Change to `evaluation` directory

   ```
   cd evaluation
   ```

2. Execute the `eval.py` Python script to evaluate the inference results of LLMs and generate metrics for them. A csv table will be generated to compare the metrics.

   ```sh
   python eval.py
   ```


# Characteristics Analysis of Code with Different Types

1. Change to `characteristic_analysis/data`, execute `gen_json.py` to process raw data.

   ```sh
   cd characteristic_analysis/data
   python gen_json.py
   ```

2. Change back to `characteristic_analysis`, execute `analyze.py` Python script. The analyze result will be stored in `characteristic_analysis/results` directory.

   ```sh
   cd ../
   python analyze.py
   ```

   The analysis may take a long time (especially the naturalness analysis). You may skip certain stage by modifying the function `analyze_data`.

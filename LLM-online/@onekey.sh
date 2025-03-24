HF_HUB_OFFLINE=1

python llm_online.py -u https://api.chatanywhere.tech/v1/ -k [api_key] -m gpt-3.5-turbo -d pseudo_code -t 16384
python llm_online.py -u https://api.chatanywhere.tech/v1/ -k [api_key] -m gpt-3.5-turbo -d assembly_code -t 16384

python llm_online.py -u https://api.deepseek.com/v1 -k [deepseek_api_key] -m deepseek-reasoner -d pseudo_code -t 65536
python llm_online.py -u https://api.deepseek.com/v1 -k [deepseek_api_key] -m deepseek-reasoner -d assembly_code -t 65536

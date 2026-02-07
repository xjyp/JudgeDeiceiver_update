import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = True
    config.stop_on_success = True
    config.tokenizer_paths = [
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "meta-llama/Meta-Llama-3-8B-Instruct"
   ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False, "temperature": 0.0}
    ]
        # 使用相对路径定位到仓库根目录下的本地模型目录
    # repo_root = <repo>/  -> models/Meta-Llama-3-8B-Instruct
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_dir = os.path.join(repo_root, "models", "Meta-Llama-3-8B-Instruct")

    config.tokenizer_paths = [local_dir]
    config.model_paths = [local_dir]
    config.tokenizer_kwargs = [{"use_fast": False, "local_files_only": True}]
    config.model_kwargs = [{
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "temperature": 0.0,
        "local_files_only": True
    }]
    
    config.conversation_templates = ["llama-3"]
    config.devices = ["cuda:0"]

    return config

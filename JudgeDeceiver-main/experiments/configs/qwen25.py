import os

os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = True
    config.stop_on_success = True

    # Use a local Qwen3-8B-Instruct model directory under repo_root/models
    # repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # local_dir = os.path.join(repo_root, "models", "Qwen3-8B")
    local_dir = "/share/disk/llm_cache/Qwen2.5-7B-Instruct"
    config.tokenizer_paths = [local_dir]
    config.model_paths = [local_dir]
    config.tokenizer_kwargs = [{"use_fast": False, "local_files_only": True}]
    config.model_kwargs = [{
        "low_cpu_mem_usage": True,
        "use_cache": False,
        "temperature": 0.0,
        "local_files_only": True
    }]
    config.conversation_templates = ["qwen"]
    config.devices = ["cuda:0"]
    config.batch_size = 16
    config.judge_model = local_dir
    return config
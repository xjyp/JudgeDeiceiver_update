from ml_collections import config_dict
# 在这个模块中定义攻击方式
def get_config():
    config = config_dict.ConfigDict()

    # General parameters 
    config.target_weight=1.0
    config.align_weight = 1.0
    config.enhance_weight=1.0
    config.perplexity_weight=0.1
    config.control_weight=0.1
    config.progressive_goals=False 
    config.progressive_models=False
    config.anneal=False
    config.incr_control=False
    config.stop_on_success=False
    config.verbose=True
    config.allow_non_ascii=False
    config.num_train_models=1

    # Results
    config.result_prefix = '../results/test'
    # 使用 Qwen3-8B 作为攻击模型（本地目录 models/Qwen3-8B）
    # tokenizers
    config.tokenizer_paths=['/root/autodl-tmp/Qwen3-8B']
    config.tokenizer_kwargs=[{"use_fast": False, "local_files_only": True}]

    config.model_paths=['/root/autodl-tmp/Qwen3-8B']
    config.model_kwargs=[{"low_cpu_mem_usage": True, "use_cache": False, "local_files_only": True}]
    config.conversation_templates=['qwen']
    config.devices=['cuda:0']

    # data
    config.train_data = ''
    config.test_data = ''
    config.n_train_data = 6
    config.n_test_data = 6
    config.data_offset = 0

    # attack-related parameters，进行攻击方式的定义
    config.attack = 'gcg'
    # 缩短控制串以降低梯度计算时的显存占用
    # config.control_init = "correct correct correct correct correct correct"
    config.control_init = "correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct correct"
    # 缩短控制串以降低梯度计算时的显存占用
    config.control_init = "correct correct correct correct correct correct"
    config.n_steps = 10
    config.test_steps = 1
    config.batch_size = 256

    config.n_steps = 10
    config.test_steps = 1
    config.batch_size = 256
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    config.filter_cand = True

    config.gbda_deterministic = True

    #eval parameters
    config.eval_model = 'qwen3'

    return config

直接下载tap.py替换即可或者下载tap_reproduce运行

# JudgeDeiceiver
这个项目主要跑Judgedeceiver/JudgeDeceiver-main/experiments/basic_scripts/下面的run_attack.sh

直接bash run1_baseline_tap.sh

一、模型路径修改
1、更换模型主要修改这个路径
JUDGE_MODEL_PATH="/root/autodl-tmp/Qwen2.5-7B-Instruct"

<img width="711" height="511" alt="ad5f48bdbbbd4f45660bb369440a6898" src="https://github.com/user-attachments/assets/d07e50da-0832-4384-9681-a830c6d93fc1" />
2、在这里进行相关参数的配置
2.1 gemma31b主要是指模型的配置文件的名字，需要同步在
<img width="1131" height="1039" alt="image" src="https://github.com/user-attachments/assets/1ef290f9-6e90-4b4d-9a72-c961b377fa53" />



三、环境配置
conda create -n Zhouzheng python=3.12

source /etc/network_turbo

pip install -r requirements.txt





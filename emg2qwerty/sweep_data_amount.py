import os
import yaml
import subprocess
import time

# ===== 核心配置 =====
CONCURRENT_JOBS = 6  # RTX 5090

# 原始的 16 个 Train Session
ALL_TRAIN_SESSIONS = [
    {"user": 89335547, "session": "2021-06-03-1622765527-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-02-1622681518-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-04-1622863166-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-07-22-1627003020-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-07-21-1626916256-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-07-22-1627004019-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-05-1622885888-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-02-1622679967-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-03-1622764398-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-07-21-1626917264-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-05-1622889105-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-03-1622766673-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-04-1622861066-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-07-22-1627001995-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-06-05-1622884635-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
    {"user": 89335547, "session": "2021-07-21-1626915176-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"},
]

VAL_SESSIONS = [{"user": 89335547, "session": "2021-06-04-1622862148-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"}]
TEST_SESSIONS = [{"user": 89335547, "session": "2021-06-02-1622682789-keystrokes-dca-study@1-0efbe614-9ae6-4131-9192-4398359b4f5f"}]

DATA_SCALES = {
    "75": 12,  # 16 * 0.75
    "50": 8,   # 16 * 0.50
    "25": 4    # 16 * 0.25
}

def generate_yaml_configs():
    print("=" * 75)
    print("🗂️ 正在自动生成数据量切分 YAML 配置文件...")
    
    # 兼容路径（不管你是在根目录还是在 emg2qwerty 目录下运行）
    config_dir = "config/user" if os.path.exists("config/user") else "emg2qwerty/config/user"
    os.makedirs(config_dir, exist_ok=True)
    
    for pct, num_sessions in DATA_SCALES.items():
        yaml_content = {
            "user": f"single_user_{pct}",
            "dataset": {
                "train": ALL_TRAIN_SESSIONS[:num_sessions],
                "val": VAL_SESSIONS,
                "test": TEST_SESSIONS
            }
        }
        
        file_path = os.path.join(config_dir, f"single_user_{pct}.yaml")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# @package _global_\n") # Hydra 要求的全局注入
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"✅ 生成成功: {file_path} (包含 {num_sessions} 个 Train Sessions)")
    print("=" * 75 + "\n")

# 🏆 Top 3 冠军阵容 (与之前完全对齐)
champions = [
    {"model": "tds_conv_ctc", "wl": 4000, "hop": 16},
    {"model": "tds_conv_ctc", "wl": 6000, "hop": 16},
    {"model": "tds_conv_ctc", "wl": 8000, "hop": 16},
    {"model": "cnn_lstm_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "cnn_lstm_ctc", "wl": 4000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "cnn_lstm_ctc", "wl": 6000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "cnn_rnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "crnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.5},
    {"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.25},
    {"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "gru_ctc", "wl": 8000, "hop": 64, "nl": 3, "hs": 512, "dp": 0.5},
    {"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 512, "dp": 0.5},
    {"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 256, "dp": 0.25},
    {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.25},
    {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.75},
    {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 5, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 2, "hs": 512, "dp": 0.25, "act": "relu"},
    {"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25},
    {"model": "tcn_ctc", "wl": 4000, "hop": 32, "dp": 0.25},
    {"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25}, 
]

def launch(cmd):
    print(f"\n🚀 Launching:\n  {cmd}\n")
    return subprocess.Popen(cmd, shell=True)

def main():
    # 1. 首先自动生成 YAML
    generate_yaml_configs()

    # 2. 将数据规模与模型组合起来
    # 这会产生 3 个数据规模 * 23 个模型 = 69 个实验
    sweep_tasks = []
    for pct in DATA_SCALES.keys():
        for config in champions:
            sweep_tasks.append((pct, config))

    total = len(sweep_tasks)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    
    print("=" * 75)
    print(f"📊 DATA SCALING ABLATION SWEEP (75%, 50%, 25%) 📊")
    print(f"Total Experiments: {total} (23 models x 3 scales)")
    print(f"Concurrent jobs: {CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")
    print("=" * 75)
    
    for i in range(0, total, CONCURRENT_JOBS):
        batch = sweep_tasks[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1
        print(f"\n{'='*25} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*25}\n")

        processes = []
        for pct, config in batch:
            model_name = config['model']
            
            # 🎯 自动判断 Epochs：双模型叠加给 100，单体模型给 80
            hybrid_models = ["cnn_lstm_ctc", "cnn_rnn_ctc", "crnn_ctc", "cnn_transformer_ctc"]
            epochs = 100 if model_name in hybrid_models else 80

            # 基础启动命令：通过 user=... 完美挂载我们刚生成的 YAML
            cmd = (
                f"python -m emg2qwerty.train "
                f"user=single_user_{pct} "  # 🎯 这里是魔法的核心！
                f"model={model_name} "
                f"datamodule.window_length={config['wl']} "
                f"logspec.hop_length={config['hop']} "
                f"trainer.accelerator=gpu "
                f"trainer.devices=1 "
                f"trainer.max_epochs={epochs} " # 动态 Epoch
                f"seed=1501 "
            )
            
            # 附加模型专属参数
            if "nl" in config: cmd += f"module.num_layers={config['nl']} "
            if "hs" in config: cmd += f"module.hidden_size={int(config['hs'])} "
            if "dp" in config: cmd += f"module.dropout={config['dp']} "
            if "act" in config: cmd += f"module.nonlinearity={config['act']} "
            
            p = launch(cmd)
            processes.append(p)
            time.sleep(3) # 错峰启动防止 IO 阻塞

        print(f"\n⏳ Waiting for Round {round_id} to finish...")
        for p in processes:
            p.wait()
        print(f"✅ Round {round_id} complete!\n")

    print("\n🎉🎉🎉 数据量消融实验全部跑完！所有的 Log 都在各自的日期文件夹里！\n")

if __name__ == "__main__":
    main()
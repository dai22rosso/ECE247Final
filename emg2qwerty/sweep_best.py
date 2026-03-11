import subprocess
import time

# ===== Fixed parameters =====
EPOCHS = 500       
CONCURRENT_JOBS = 6  # 适配你的 RTX 5090，如果遇到 OOM 随时改成 3 或 4

def launch(cmd):
    print(f"\n🚀 Launching:\n  {cmd}\n")
    return subprocess.Popen(cmd, shell=True)

# 🏆 严格基于你提供的 CSV 数据提取的 Top 参数 (绝对没有任何瞎编/添加)
champions = [
    # --- 1. TDSConvCTCModule (映射为你的 model=tds_conv_ctc) ---
    {"model": "tds_conv_ctc", "wl": 4000, "hop": 16},
    {"model": "tds_conv_ctc", "wl": 6000, "hop": 16},
    {"model": "tds_conv_ctc", "wl": 8000, "hop": 16},

    # --- 2. cnn_lstm_ctc ---
    {"model": "cnn_lstm_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "cnn_lstm_ctc", "wl": 4000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
    {"model": "cnn_lstm_ctc", "wl": 6000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5},

    # --- 3. cnn_rnn_ctc ---
    {"model": "cnn_rnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"},

    # --- 4. crnn_ctc (全场冠军起跑线) ---
    {"model": "crnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.5},
    {"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.25},
    {"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},

    # --- 5. gru_ctc ---
    {"model": "gru_ctc", "wl": 8000, "hop": 64, "nl": 3, "hs": 512, "dp": 0.5},
    {"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 512, "dp": 0.5},
    {"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 256, "dp": 0.25},

    # --- 6. lstm_ctc ---
    {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.25},
    {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.75},
    {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5},

    # --- 7. rnn_ctc ---
    # 注：你在 CSV 中的 rnn_ctc 有两行的参数是完全一模一样的重复，这里我只保留了两组独立的参数
    {"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 5, "hs": 256, "dp": 0.25, "act": "relu"},
    {"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 2, "hs": 512, "dp": 0.25, "act": "relu"},

    # --- 8. tcn_ctc ---
    # 严格按照 CSV 提供的内容 (只有 Dropout, Window_Len 和 Hop_Len，不再画蛇添足)
    {"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25},
    {"model": "tcn_ctc", "wl": 4000, "hop": 32, "dp": 0.25},
    # 你的 CSV 中 TCN 也有两组参数是重复的 (wl=8000, hop=32, dp=0.25)，为了跑够数我依然放了3组
    {"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25}, 
]

def main():
    total = len(champions)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    
    print("=" * 75)
    print(f"🏆 STRICT TOP-3 CHAMPIONS SWEEP (500 EPOCHS) 🏆")
    print(f"Total Elite Models: {total}")
    print(f"Concurrent jobs: {CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")
    print("=" * 75)
    
    for i in range(0, total, CONCURRENT_JOBS):
        batch = champions[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1
        print(f"\n{'='*25} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*25}\n")

        processes = []
        for config in batch:
            # 基础启动命令：像以前的 Sweep 一样纯净
            cmd = (
                f"python -m emg2qwerty.train "
                f"user=single_user "
                f"model={config['model']} "
                f"datamodule.window_length={config['wl']} "
                f"logspec.hop_length={config['hop']} "
                f"trainer.accelerator=gpu "
                f"trainer.devices=1 "
                f"trainer.max_epochs={EPOCHS} "
                f"seed=1501 "
            )
            
            # 只有 CSV 里显式写了的参数，才会被传进命令行
            if "nl" in config: cmd += f"module.num_layers={config['nl']} "
            if "hs" in config: cmd += f"module.hidden_size={int(config['hs'])} "
            if "dp" in config: cmd += f"module.dropout={config['dp']} "
            if "act" in config: cmd += f"module.nonlinearity={config['act']} "
            
            p = launch(cmd)
            processes.append(p)
            time.sleep(3) # 错峰启动，防止同时读文件卡死

        print(f"\n⏳ Waiting for Round {round_id} to finish...")
        for p in processes:
            p.wait()
        print(f"✅ Round {round_id} complete!\n")

    print("\n🎉🎉🎉 500-Epoch 终极大长跑全部结束！所有的 Log 都在各自的日期文件夹里！\n")

if __name__ == "__main__":
    main()
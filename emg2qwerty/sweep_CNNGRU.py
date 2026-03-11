import itertools
import subprocess
import time

# ===== Fixed parameters =====
BIDIRECTIONAL = "true"
EPOCHS = 100       # EarlyStopping 会自动切断烂尾实验

# ===== CRNN Search space =====
num_layers_opts = [1, 2, 3]                 # GRU 层数 (CNN 已打底，无需过深)
hidden_size_opts = [256, 512]            # GRU 隐层维度 (256轻量级 vs 512冠军级)
dropout_opts = [0.25, 0.5]               # 正则化区间
window_length_opts = [4000, 6000, 8000]  # 感受野窗口探测
hop_length_opts = [16, 32, 64]           # 🎯 核心冲突点：CNN(16) 与 GRU(64) 的博弈

# ===== Concurrency =====
CONCURRENT_JOBS = 6  # 适配 RTX 5090 的显存

def launch(cmd):
    print(f"\n🚀 Launching:\n  {cmd}\n")
    return subprocess.Popen(cmd, shell=True)

def main():
    combos = list(itertools.product(
        num_layers_opts,
        hidden_size_opts,
        dropout_opts,
        window_length_opts,
        hop_length_opts, # 加入 hop_length
    ))

    total = len(combos)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    print("=" * 70)
    print(f"🔥 CRNN (CNN + GRU) Ultimate Sweep 🔥")
    print(f"Fixed: bidirectional={BIDIRECTIONAL}, epochs={EPOCHS}")
    print(f"Search: layers={num_layers_opts}, hidden={hidden_size_opts}")
    print(f"        dropout={dropout_opts}, window_len={window_length_opts}")
    print(f"        hop_length={hop_length_opts} 🎯")
    print(f"Total experiments: {total}")
    print(f"Concurrent jobs: {CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")
    print("=" * 70)

    for i in range(0, total, CONCURRENT_JOBS):
        batch = combos[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1
        print(f"\n{'='*25} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*25}\n")

        processes = []
        for nl, hs, dp, wl, ho in batch:
            cmd = (
                "python -m emg2qwerty.train "
                "user=single_user "
                "model=crnn_ctc "
                f"module.bidirectional={BIDIRECTIONAL} "
                f"module.num_layers={nl} "
                f"module.hidden_size={hs} "
                f"module.dropout={dp} "
                f"datamodule.window_length={wl} "
                f"logspec.hop_length={ho} "  # 动态传入 hop_length
                "trainer.accelerator=gpu "
                "trainer.devices=1 "
                f"trainer.max_epochs={EPOCHS} "
                "seed=1501"
            )
            p = launch(cmd)
            processes.append(p)
            # 防止 IO 冲突
            time.sleep(3)

        print(f"\n⏳ Waiting for Round {round_id} to finish...")
        for p in processes:
            p.wait()
        print(f"✅ Round {round_id} complete!\n")

    print("\n🎉🎉🎉 ALL CRNN EXPERIMENTS FINISHED! 看看 CNN 和 GRU 到底谁向谁妥协了！\n")

if __name__ == "__main__":
    main()
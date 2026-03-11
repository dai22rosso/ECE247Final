import itertools
import subprocess
import time

# ===== Fixed parameters =====
EPOCHS = 80       

# ===== TCN Search space =====
# 注意：在 Hydra 命令行中传列表，需要加上引号
channels_opts = ["[128,128,128,128]", "[256,256,256,256]", "[512,512,512,512]"]  # 增加 128 轻量级
kernel_size_opts = [3, 5]                                   
causal_opts = ["false", "true"]                             # 🎯 核心消融实验：因果性
window_length_opts = [4000,  8000]                     
hop_length_opts = [16, 32]                                  # 🎯 分辨率博弈 (16 高清 vs 32 降采样)
dropout_opts = [0.0, 0.25, 0.5]                             # 🎯 正则化测试 (0.0 用来验证是否过拟合)

# ===== Concurrency =====
CONCURRENT_JOBS = 6  # 适配你的 RTX 5090

def launch(cmd):
    print(f"\n🚀 Launching:\n  {cmd}\n")
    return subprocess.Popen(cmd, shell=True)

def main():
    combos = list(itertools.product(
        channels_opts,
        kernel_size_opts,
        causal_opts,
        window_length_opts,
        hop_length_opts,
        dropout_opts
    ))

    total = len(combos)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    print("=" * 75)
    print(f"🔥 Acausal vs Causal TCN Ultimate Sweep 🔥")
    print(f"Fixed: epochs={EPOCHS}")
    print(f"Search: channels={channels_opts}")
    print(f"        kernel={kernel_size_opts}, causal={causal_opts}")
    print(f"        window_len={window_length_opts}, hop_length={hop_length_opts}")
    print(f"        dropout={dropout_opts}")
    print(f"Total experiments: {total}")
    print(f"Concurrent jobs: {CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")
    print("=" * 75)

    for i in range(0, total, CONCURRENT_JOBS):
        batch = combos[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1
        print(f"\n{'='*25} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*25}\n")

        processes = []
        # 注意：这里解包参数变成了 6 个
        for ch, ks, csl, wl, ho, dp in batch:
            cmd = (
                "python -m emg2qwerty.train "
                "user=single_user "
                "model=tcn_ctc "  # 🎯 指向 config/model/tcn_ctc.yaml
                f"module.num_channels='{ch}' "  
                f"module.kernel_size={ks} "
                f"module.causal={csl} "
                f"module.dropout={dp} "
                f"datamodule.window_length={wl} "
                f"logspec.hop_length={ho} "
                "trainer.accelerator=gpu "
                "trainer.devices=1 "
                f"trainer.max_epochs={EPOCHS} "
                "seed=1501"
            )
            p = launch(cmd)
            processes.append(p)
            # TCN 初始化极快，停顿 2 秒防止 IO 踩踏即可
            time.sleep(2)

        print(f"\n⏳ Waiting for Round {round_id} to finish...")
        for p in processes:
            p.wait()
        print(f"✅ Round {round_id} complete!\n")

    print("\n🎉🎉🎉 ALL TCN EXPERIMENTS FINISHED! 准备好迎接震撼结果吧！\n")

if __name__ == "__main__":
    delay_hours = 5
    delay_seconds = delay_hours * 3600
    print("=" * 75)
    print(f"⏰ Timer set! Waiting for {delay_hours} hours before starting the sweep...")
    print(f"⏰ The sweep will automatically start in {delay_seconds} seconds.")
    print("=" * 75)
    
    # 🎯 核心：让脚本在这里睡 5 个小时
    time.sleep(delay_seconds)
    print("\n🚀 Waking up! Starting the TCN Ultimate Sweep now...\n")
    main()
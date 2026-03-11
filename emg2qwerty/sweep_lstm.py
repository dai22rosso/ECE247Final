import itertools
import subprocess
import time

# 1. 定义你要搜索的超参数空间
bidirectional_opts = ["true", "false"]  # Hydra 识别 true/false
num_layers_opts = [1, 2, 3, 5]
hidden_size_opts = [128, 256, 512]
dropout_opts = [0.25, 0.5, 0.75]
epochs = [80]
# 2. 每次并行跑几个实验？（你的显卡能抗住的数量）
CONCURRENT_JOBS = 8

def main():
    # 生成 72 种全排列组合
    all_combinations = list(itertools.product(
        bidirectional_opts, num_layers_opts, hidden_size_opts, dropout_opts, epochs
    ))
    
    total_experiments = len(all_combinations)
    print(f"🔥 共生成 {total_experiments} 组实验配置，每次并行跑 {CONCURRENT_JOBS} 组。")
    print(f"🔥 预计分为 {total_experiments // CONCURRENT_JOBS + (1 if total_experiments % CONCURRENT_JOBS else 0)} 轮进行。\n")

    # 分块（Chunks）执行
    for i in range(0, total_experiments, CONCURRENT_JOBS):
        batch = all_combinations[i : i + CONCURRENT_JOBS]
        round_num = (i // CONCURRENT_JOBS) + 1
        print(f"========== 🚀 开始第 {round_num} 轮训练 (包含 {len(batch)} 个实验) ==========")
        
        processes = []
        for combo in batch:
            bi, nl, hs, dp ,ep= combo
            
            # 组装 Hydra 训练命令
            # 注意：如果是单层 LSTM，我们依然传 dropout，PyTorch 会自动忽略它并给个 Warning，不影响训练
            cmd = (
                f"python -m emg2qwerty.train user=single_user model=lstm_ctc "
                f"module.bidirectional={bi} module.num_layers={nl} "
                f"module.hidden_size={hs} module.dropout={dp} "
                f"trainer.max_epochs={ep} "
                f"trainer.accelerator=gpu trainer.devices=1"
            )
            
            print(f"[启动] -> {cmd}")
            # subprocess.Popen 会在后台默默启动进程，不会阻塞当前的 Python 脚本
            p = subprocess.Popen(cmd, shell=True)
            processes.append(p)
            
            # 稍微停顿 2 秒，防止多个进程瞬间同时读取 HDF5 文件导致 IO 拥堵报错
            time.sleep(2)
        
        # 核心逻辑：等待这 8 个进程全部运行完毕，再进入下一个 for 循环
        print(f"⏳ 第 {round_num} 轮实验已全部在后台启动，等待它们执行完毕...")
        for p in processes:
            p.wait()
            
        print(f"✅ 第 {round_num} 轮实验结束！\n")

    print("🎉🎉🎉 恭喜！所有 72 组实验已全部跑完！请去 outputs/ 目录查看战果！")

if __name__ == "__main__":
    main()
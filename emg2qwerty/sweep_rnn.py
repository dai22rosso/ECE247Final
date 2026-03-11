import itertools
import subprocess
import time

# =========================
# Sweep Search Space (32)
# =========================

bidirectional_opts = ["true"]
num_layers_opts = [2,3,4,5,6,7,8]
hidden_size_opts = [256, 128,512, 64, 1024]
dropout_opts = [0.25]
nonlinearity_opts = [ "relu"]

hop_length = [64,128]
epochs = 60

# 并行数量（按 GPU 数量调）
CONCURRENT_JOBS = 6


def launch(cmd):
    print(f"\n🚀 Launching:\n{cmd}\n")
    return subprocess.Popen(cmd, shell=True)


def main():

    combos = list(itertools.product(
        bidirectional_opts,
        num_layers_opts,
        hidden_size_opts,
        dropout_opts,
        nonlinearity_opts,
        hop_length,
    ))

    print("====================================")
    print(f"Total Experiments: {len(combos)}")
    print(f"Concurrent Jobs: {CONCURRENT_JOBS}")
    print("====================================")

    for i in range(0, len(combos), CONCURRENT_JOBS):

        batch = combos[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1

        print(f"\n========== Round {round_id} ==========\n")

        processes = []

        for combo in batch:

            bi, nl, hs, dp, nonlin ,ho= combo

            cmd = (
                "python -m emg2qwerty.train "
                "user=single_user "
                "model=rnn_ctc "
                f"module.bidirectional={bi} "
                f"module.num_layers={nl} "
                f"module.hidden_size={hs} "
                f"module.dropout={dp} "
                f"module.nonlinearity={nonlin} "
                f"logspec.hop_length={ho} "
                "trainer.accelerator=gpu "
                "trainer.devices=1 "
                f"trainer.max_epochs={epochs} "
                "seed=1501"
            )

            p = launch(cmd)
            processes.append(p)

            time.sleep(2)

        print("\n⏳ Waiting batch to finish...\n")

        for p in processes:
            p.wait()

    print("\n🎉 ALL RNN EXPERIMENTS FINISHED\n")


if __name__ == "__main__":
    main()
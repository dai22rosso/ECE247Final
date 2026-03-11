
import itertools
import subprocess
import time

# ===== Fixed parameters =====
BIDIRECTIONAL = "true"
EPOCHS = 80  # EarlyStopping will cut bad runs early

# ===== Search space =====
num_layers_opts = [1, 2, 3, 4, 5]           # LSTM only works at 2; RNN works at 2-5; GRU try 1-4
hidden_size_opts = [128, 256, 512]       # LSTM sweet spot 512; RNN sweet spot 256-512
dropout_opts = [0.1, 0.25, 0.5]         # 0.25 is universal winner, bracket it
hop_length_opts = [16, 32, 64]              # LSTM best at 16, RNN best at 64, test both

# ===== Concurrency =====
CONCURRENT_JOBS = 6  # RTX 5090

def launch(cmd):
    print(f"\n🚀 Launching:\n  {cmd}\n")
    return subprocess.Popen(cmd, shell=True)


def main():
    combos = list(itertools.product(
        num_layers_opts,
        hidden_size_opts,
        dropout_opts,
        hop_length_opts,
    ))

    total = len(combos)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    print("=" * 60)
    print(f"GRU Sweep — Informed by LSTM & RNN v2")
    print(f"Fixed: bidirectional={BIDIRECTIONAL}, epochs={EPOCHS}")
    print(f"Search: layers={num_layers_opts}, hidden={hidden_size_opts}")
    print(f"        dropout={dropout_opts}, hop={hop_length_opts}")
    print(f"Total experiments: {total}")
    print(f"Concurrent jobs: {CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")
    print("=" * 60)

    for i in range(0, total, CONCURRENT_JOBS):
        batch = combos[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1
        print(f"\n{'='*20} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*20}\n")

        processes = []
        for nl, hs, dp, ho in batch:
            cmd = (
                "python -m emg2qwerty.train "
                "user=single_user "
                "model=gru_ctc "
                f"module.bidirectional={BIDIRECTIONAL} "
                f"module.num_layers={nl} "
                f"module.hidden_size={hs} "
                f"module.dropout={dp} "
                f"logspec.hop_length={ho} "
                "trainer.accelerator=gpu "
                "trainer.devices=1 "
                f"trainer.max_epochs={EPOCHS} "
                "seed=1501"
            )
            p = launch(cmd)
            processes.append(p)
            time.sleep(2)

        print(f"\n⏳ Waiting for Round {round_id} to finish...")
        for p in processes:
            p.wait()
        print(f"✅ Round {round_id} complete!\n")

    print("\n🎉 ALL GRU EXPERIMENTS FINISHED!\n")


if __name__ == "__main__":
    main()
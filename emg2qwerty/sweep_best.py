import subprocess
import time

EPOCHS = 100       
CONCURRENT_JOBS = 6 

def launch(cmd):
    print(f"\n🚀 Launching:\n  {cmd}\n")
    return subprocess.Popen(cmd, shell=True)

champions = [

]

def main():
    total = len(champions)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    print(f"Total Elite Models: {total}")
    print(f"Concurrent jobs: {CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")
    
    for i in range(0, total, CONCURRENT_JOBS):
        batch = champions[i:i + CONCURRENT_JOBS]
        round_id = i // CONCURRENT_JOBS + 1
        print(f"\n{'='*25} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*25}\n")

        processes = []
        for config in batch:
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
            if "nl" in config: cmd += f"module.num_layers={config['nl']} "
            if "hs" in config: cmd += f"module.hidden_size={int(config['hs'])} "
            if "dp" in config: cmd += f"module.dropout={config['dp']} "
            if "act" in config: cmd += f"module.nonlinearity={config['act']} "
            
            p = launch(cmd)
            processes.append(p)
            time.sleep(3) 
        for p in processes:
            p.wait()
        print(f"Round {round_id} complete!\n")

    print("\nDone\n")

if __name__ == "__main__":
    main()
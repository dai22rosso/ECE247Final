import os
import yaml
import subprocess
import time

CONCURRENT_JOBS = 6  # RTX 5090
LOGSPEC_TRANSFORMS = [
    "log_spectrogram",
    # "log_spectrogram3",
    # "log_spectrogram4",
    # "log_spectrogram5",
    # "log_spectrogram6",
    # "log_spectrogram7",
]

WINDOW_LENGTHS = [2000, 4000, 6000, 8000, 12000, 16000]
# WINDOW_LENGTHS = [ 8000]

EXTRA_CONCURRENT_JOBS = 7


def run_logspec_window_sweep():
    sweep_tasks = []
    for tf in LOGSPEC_TRANSFORMS:
        for wl in WINDOW_LENGTHS:
            sweep_tasks.append((tf, wl))

    total = len(sweep_tasks)
    rounds = (total + EXTRA_CONCURRENT_JOBS - 1) // EXTRA_CONCURRENT_JOBS

    print(f"Total Experiments: {total}")
    print(f"Concurrent jobs: {EXTRA_CONCURRENT_JOBS}")
    print(f"Estimated rounds: {rounds}")


    for i in range(0, total, EXTRA_CONCURRENT_JOBS):
        batch = sweep_tasks[i:i + EXTRA_CONCURRENT_JOBS]
        round_id = i // EXTRA_CONCURRENT_JOBS + 1
        print(f"\n{'='*25} Round {round_id}/{rounds} ({len(batch)} jobs) {'='*25}\n")

        processes = []

        for tf, wl in batch:
            cmd = (
                f"python -m emg2qwerty.train "
                f"transforms={tf} "
                f"user=single_user "
                f"datamodule.window_length={wl} "
                f"trainer.accelerator=gpu "
                f"trainer.devices=1 "
                f"seed=1501 "
            )

            p = launch(cmd)
            processes.append(p)
            time.sleep(2)
        for p in processes:
            p.wait()

        print(f"✅ Round {round_id} complete!\n")

    print("\nWindow length Done\n")
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
    "75": 12,  
    "50": 8,   
    "25": 4   
}

def generate_yaml_configs():
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
            f.write("# @package _global_\n")
            yaml.dump(yaml_content, f, sort_keys=False)
        print(f"✅ 生成成功: {file_path} (包含 {num_sessions} 个 Train Sessions)")
    print("=" * 75 + "\n")

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
    generate_yaml_configs()
    sweep_tasks = []
    for pct in DATA_SCALES.keys():
        for config in champions:
            sweep_tasks.append((pct, config))

    total = len(sweep_tasks)
    rounds = (total + CONCURRENT_JOBS - 1) // CONCURRENT_JOBS
    
    print("=" * 75)
    print(f"DATA SCALING ABLATION SWEEP (75%, 50%, 25%) ")
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
            hybrid_models = ["cnn_lstm_ctc", "cnn_rnn_ctc", "crnn_ctc", "cnn_transformer_ctc"]
            epochs = 100 if model_name in hybrid_models else 100
            cmd = (
                f"python -m emg2qwerty.train "
                f"user=single_user_{pct} " 
                f"model={model_name} "
                f"datamodule.window_length={config['wl']} "
                f"logspec.hop_length={config['hop']} "
                f"trainer.accelerator=gpu "
                f"trainer.devices=1 "
                f"trainer.max_epochs={epochs} " 
                f"seed=1501 "
            )
            
            if "nl" in config: cmd += f"module.num_layers={config['nl']} "
            if "hs" in config: cmd += f"module.hidden_size={int(config['hs'])} "
            if "dp" in config: cmd += f"module.dropout={config['dp']} "
            if "act" in config: cmd += f"module.nonlinearity={config['act']} "
            
            p = launch(cmd)
            processes.append(p)
            time.sleep(3) 

        print(f"\n⏳ Waiting for Round {round_id} to finish...")
        for p in processes:
            p.wait()
        print(f"✅ Round {round_id} complete!\n")

    print("\n Ablation Done\n")

if __name__ == "__main__":
    delay_hours = 0
    delay_seconds = delay_hours * 3600
    print("=" * 75)
    print(f" Timer set! Waiting for {delay_hours} hours before starting the sweep...")
    print(f" The sweep will automatically start in {delay_seconds} seconds.")
    print("=" * 75)
    time.sleep(delay_seconds)
    main()
    run_logspec_window_sweep()
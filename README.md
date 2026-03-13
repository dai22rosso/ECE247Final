# C147/247 Final Project
### Winter 2026 
This is the final project for ECE247 course at UCLA.
## Architecture

# CNN
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 
# Others except CNN+Transformer
cnn_lstm_ctc
  {"model": "cnn_lstm_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
  {"model": "cnn_lstm_ctc", "wl": 4000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},
  {"model": "cnn_lstm_ctc", "wl": 6000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5},

cnn_rnn_ctc
  {"model": "cnn_rnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"},
  {"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 256, "dp": 0.25, "act": "relu"},
  {"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"},

cnn_gru_ctc
  {"model": "crnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.5},
  {"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.25},
  {"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5},

gru_ctc
  {"model": "gru_ctc", "wl": 8000, "hop": 64, "nl": 3, "hs": 512, "dp": 0.5},
  {"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 512, "dp": 0.5},
  {"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 256, "dp": 0.25},

lstm_ctc
  {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.25},
  {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.75},
  {"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5},

rnn_ctc
  {"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 5, "hs": 256, "dp": 0.25, "act": "relu"},
  {"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 2, "hs": 512, "dp": 0.25, "act": "relu"},

tcn_ctc
  {"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25},
  {"model": "tcn_ctc", "wl": 4000, "hop": 32, "dp": 0.25},
  {"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25}, 
  
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
Fill in the blank when you have them.

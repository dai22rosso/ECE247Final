# C147/247 Final Project

**Winter 2026**  
Final project for **ECE C147/C247** at **UCLA**.

---

## Training Overview

All experiments are run with the following common settings unless otherwise specified:

- `user=single_user`
- `trainer.accelerator=gpu`
- `trainer.devices=1`
- `seed=1501`

Most default settings are defined in `modules.py` and `lightning.py`, so no source-code changes are needed for standard experiment runs unless explicitly stated.

---

# 1. Architecture Experiments

## 1.1 CNN

Base command:

```bash
python -m emg2qwerty.train \
  user=single_user \
  trainer.accelerator=gpu \
  trainer.devices=1
```

---

## 1.2 Other Architectures (except CNN+Transformer)

### CNN + LSTM (`cnn_lstm_ctc`)

```python
{"model": "cnn_lstm_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5}
{"model": "cnn_lstm_ctc", "wl": 4000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5}
{"model": "cnn_lstm_ctc", "wl": 6000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5}
```

### CNN + RNN (`cnn_rnn_ctc`)

```python
{"model": "cnn_rnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"}
{"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 2, "hs": 256, "dp": 0.25, "act": "relu"}
{"model": "cnn_rnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 256, "dp": 0.25, "act": "relu"}
```

### CNN + GRU (`crnn_ctc`)

```python
{"model": "crnn_ctc", "wl": 8000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.5}
{"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 3, "hs": 512, "dp": 0.25}
{"model": "crnn_ctc", "wl": 6000, "hop": 32, "nl": 2, "hs": 512, "dp": 0.5}
```

### GRU (`gru_ctc`)

```python
{"model": "gru_ctc", "wl": 8000, "hop": 64, "nl": 3, "hs": 512, "dp": 0.5}
{"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 512, "dp": 0.5}
{"model": "gru_ctc", "wl": 8000, "hop": 32, "nl": 5, "hs": 256, "dp": 0.25}
```

### LSTM (`lstm_ctc`)

```python
{"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.25}
{"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.75}
{"model": "lstm_ctc", "wl": 8000, "hop": 16, "nl": 2, "hs": 512, "dp": 0.5}
```

### Vanilla RNN (`rnn_ctc`)

```python
{"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 5, "hs": 256, "dp": 0.25, "act": "relu"}
{"model": "rnn_ctc", "wl": 8000, "hop": 64, "nl": 2, "hs": 512, "dp": 0.25, "act": "relu"}
```

### TCN (`tcn_ctc`)

```python
{"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25}
{"model": "tcn_ctc", "wl": 4000, "hop": 32, "dp": 0.25}
{"model": "tcn_ctc", "wl": 8000, "hop": 32, "dp": 0.25}
```

### General command template

```python
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
```

### Parameters to fill in when available

For architectures that require additional hyperparameters, add the corresponding arguments as needed:

- `module.num_layers={config['nl']}`
- `module.hidden_size={config['hs']}`
- `module.dropout={config['dp']}`
- `module.activation={config['act']}`

A more complete template looks like this:

```python
cmd = (
    f"python -m emg2qwerty.train "
    f"user=single_user "
    f"model={config['model']} "
    f"datamodule.window_length={config['wl']} "
    f"logspec.hop_length={config['hop']} "
    f"module.num_layers={config['nl']} "
    f"module.hidden_size={config['hs']} "
    f"module.dropout={config['dp']} "
    f"trainer.accelerator=gpu "
    f"trainer.devices=1 "
    f"trainer.max_epochs={EPOCHS} "
    f"seed=1501 "
)
```

For RNN-based models with activation:

```python
cmd += f"module.activation={config['act']} "
```

---

## 1.3 CNN + Transformer

### Configurations

```python
{"wl": 16000, "hop": 32, "nh": 3, "dff": 256, "dp": 0.25}
{"wl": 16000, "hop": 32, "nh": 2, "dff": 512, "dp": 0.25}
{"wl": 16000, "hop": 32, "nh": 2, "dff": 512, "dp": 0.1}
nl=1
```

### Command template

```python
cmd = (
    "python -m emg2qwerty.train "
    "user=single_user "
    "model=cnn_transformer_ctc "
    f"module.num_layers={nl} "
    f"module.nhead={nh} "
    f"module.dim_feedforward={dff} "
    f"module.dropout={dp} "
    f"datamodule.window_length={wl} "
    f"logspec.hop_length={hop} "
    f"batch_size={BATCH_SIZE} "
    "trainer.accelerator=gpu "
    "trainer.devices=1 "
    f"trainer.max_epochs={EPOCHS} "
    "seed=1501"
)
```

### Parameters to fill in later

You still need to provide:

- `nh`: number of attention heads
- `BATCH_SIZE`
- `EPOCHS`

---

# 2. Ablation Experiments

## 2.1 Data Augmentation

There are **6 additional augmentation settings** in:

```bash
./config/transform/
```

Run them with:

```python
cmd = (
    f"python -m emg2qwerty.train "
    f"transforms=log_spectrogram{i} "
    f"user=single_user "
    f"datamodule.window_length=8000 "
    f"trainer.accelerator=gpu "
    f"trainer.devices=1 "
    f"seed=1501 "
)
```

Replace `i` with the desired augmentation config index, for example `2`, `3`, ..., `7`.

Example:

```bash
python -m emg2qwerty.train \
  transforms=log_spectrogram2 \
  user=single_user \
  datamodule.window_length=8000 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  seed=1501
```

---

## 2.2 Window Length

To test different window lengths:

```python
cmd = (
    f"python -m emg2qwerty.train "
    f"user=single_user "
    f"datamodule.window_length={wl} "
    f"trainer.accelerator=gpu "
    f"trainer.devices=1 "
    f"seed=1501 "
)
```

---

## 2.3 Data Scaling

For data scaling experiments, change the user argument to:

```bash
user=single_user_{pct}
```

Examples:

```bash
user=single_user_25
user=single_user_50
user=single_user_75
user=single_user_100
```

---

## 2.4 Hop Length

For hop length ablation, modify:

```bash
logspec.hop_length={hop}
```

in the corresponding training command.

---

# 3. Notes

- All major architecture settings are already defined in `modules.py` and `lightning.py`.
- No manual code modification is required for standard experiment runs.
- Only command-line arguments need to be changed for architecture sweeps and ablations.

---



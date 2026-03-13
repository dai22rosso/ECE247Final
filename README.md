# C147/247 Final Project
### Winter 2026 
This is the final project for ECE247 course at UCLA.
## CNN
python -m emg2qwerty.train \
  user="single_user" \
  trainer.accelerator=gpu trainer.devices=1 

## Vanilla RNN
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
details seen in the ./emg2qwerty/sweep_rnn.py



## LSTM
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

# Adapter folder

This folder holds the LoRA adapter from the full training run (configs/train.yaml, 4 epochs, best validation Macro F1 0.8704, test Macro F1 0.8751).

adapter_config.json describes the LoRA setup.
adapter_model.safetensors holds the adapter weights, about 3.5 MB.
label_map.json maps ids to the three labels.
train_meta.json records the run that produced the weights, including the base model id. For the shipped weights run_name is full.

train_lora.py writes here. Running the smoke config replaces these weights with a short CPU run that only proves training finished, so check run_name in train_meta.json before quoting a score.

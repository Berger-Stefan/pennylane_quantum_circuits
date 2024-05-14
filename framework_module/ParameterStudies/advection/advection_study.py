import os, sys

# Set paths to known folder structure for different PINNs and lib files.
file_path = os.path.dirname(__file__)
parent_dir = "/".join([file_path, "../../"])
sys.path.append(parent_dir)


import logging
import optuna

from advection_wrapper import advection_config_and_training_wrapper

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "advection_equation"  # Unique identifier of the study.
storage_name = "postgresql://user:password@192.168.178.24:5432/master_thesis"

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)
# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.NopPruner(), study_name=study_name, storage=storage_name, load_if_exists=True)


study.optimize(advection_config_and_training_wrapper, n_trials=200)

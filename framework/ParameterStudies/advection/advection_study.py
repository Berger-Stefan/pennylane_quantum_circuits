import sys, os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the root directory to the Python path
sys.path.append(root_dir)

import logging
import optuna

from advection_wrapper import advection_config_and_training_wrapper

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "advection_equation"  # Unique identifier of the study.
storage_name = "postgresql://user:password@192.168.178.24:5432/master_thesis"

study = optuna.create_study(sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), study_name=study_name, storage=storage_name, load_if_exists=True)
# study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=optuna.pruners.NopPruner(), study_name=study_name, storage=storage_name, load_if_exists=True)


study.optimize(advection_config_and_training_wrapper, n_trials=200)
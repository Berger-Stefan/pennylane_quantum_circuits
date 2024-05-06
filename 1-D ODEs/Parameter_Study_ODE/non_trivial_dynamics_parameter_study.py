import sys
import logging
import optuna

from non_trivial_dynamics_wrapper import config_and_training_wrapper

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "non_trivial_dynamics"  # Unique identifier of the study.
storage_name = "postgresql://user:password@192.168.2.112:5432/{}".format(study_name)
study = optuna.create_study(sampler=optuna.samplers.TPESampler(),pruner=optuna.pruners.MedianPruner(), study_name=study_name, storage=storage_name, load_if_exists=True)



study.optimize(config_and_training_wrapper, n_trials=100)
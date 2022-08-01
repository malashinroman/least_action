import os
import numpy as np
import sys
sys.path.append(".")
from script_manager.func.script_boilerplate import do_everything
from script_manager.func.script_parse_args import get_script_args
os.environ['MKL_THREADING_LAYER'] = 'GNU'
args = get_script_args()

# weights and biases project name
wandb_project_name = "least_action"

# keys
appendix_keys = ["tag"]
extra_folder_keys = []

epochs = 200
default_parameters = {
    "action_net_hidden_size": 60,
    "baseline_loss_coeff": 1.0,
    "batch_size": 256,
    "cifar_classifier_indexes": "[0,1,2,3,4,5]",
    "dataset_type": "CIFAR-10",
    "entropy_unc_coeff": 0.01,
    "entropy_var_coeff": 50,
    "epochs": epochs,
    "gama": 1,
    "intermediate_supervision": 1,
    "load_checkpoint_path": "",
    "long_shot_baseline": 1,
    "lr_milestones": "[170,190]",
    "nll_reward": False,
    "num_glimpses": 3,
    "num_worker": 0,
    "random_seed": 1111,
    "reinforce_act": 0.01,
    "supervised_loss_coeff": 1.0,
    "use_gpu": False,
    "use_mask_state": 0,
    # "valid_size": 0.1,
    "weak_classifier_folder": "weak_classifiers/classifiers_cifar_subsets/",
}

configs = []

test_parameters = {"epochs": 3, "skip_validation": False}
main_script = "main.py"

for use_mask_state in [0]:
    for num_glimpses in range(2, 7):
        config = {
            "use_mask_state": use_mask_state,
            "num_glimpses": num_glimpses,
            "tag": f"pool1_icmlt",
            "cifar_classifier_indexes": "[0,1,2,3,4,5]",  # according to pool2 of the paper
        }
        configs.append([config, None])

# RUN everything
if __name__ == "__main__":
    do_everything(
        default_parameters=default_parameters,
        configs=configs,
        extra_folder_keys=extra_folder_keys,
        appendix_keys=appendix_keys,
        main_script=main_script,
        test_parameters=test_parameters,
        wandb_project_name=wandb_project_name,
        script_file=__file__,
    )

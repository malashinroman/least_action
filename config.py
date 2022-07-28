import argparse

from script_manager.func.add_needed_args import smart_parse_args


def str2intlist(v):
    return [int(x.strip()) for x in v.strip()[1:-1].split(",")]


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser(description="LAC")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# core network parameters
core_arg = add_argument_group("Core Network Params")
core_arg.add_argument(
    "--use_mask_state", type=str2bool, default=True, help="use_mask_state"
)

core_arg.add_argument(
    "--num_glimpses", type=int, default=6, help="# of glimpses, i.e. BPTT iterations"
)

core_arg.add_argument("--model", type=str, default="RecurrentAttention")

core_arg.add_argument("--action_net_hidden_size", type=int, default=40)

# reinforce params
reinforce_arg = add_argument_group("Reinforce Params")
reinforce_arg.add_argument("--nll_reward", type=str2bool, default=False)
reinforce_arg.add_argument("--supervised_loss_coeff", type=float, default=1.0)
reinforce_arg.add_argument("--baseline_loss_coeff", type=float, default=1.0)
reinforce_arg.add_argument("--reinforce_loss_coeff", type=float, default=0.01)
reinforce_arg.add_argument("--action_entropy_coeff", type=float, default=1e-3)
reinforce_arg.add_argument("--entropy_unc_coeff", type=float, default=0)
reinforce_arg.add_argument("--entropy_var_coeff", type=float, default=0)
reinforce_arg.add_argument("--context_sig_coeff", type=float, default=0)
reinforce_arg.add_argument("--reinforce_act_coeff", type=float, default=1.0)
reinforce_arg.add_argument("--reinforce_loc_coeff", type=float, default=1.0)
reinforce_arg.add_argument("--reinforce_stop_coeff", type=float, default=0)
reinforce_arg.add_argument("--stop_exploration_reg", type=float, default=0)
reinforce_arg.add_argument("--baseline_loss", type=str, default="mse")
reinforce_arg.add_argument("--intermediate_supervision", type=int, default=0)
reinforce_arg.add_argument("--long_shot_baseline", type=int, default=1)

# data params
data_arg = add_argument_group("Data Params")
data_arg.add_argument(
    "--valid_size",
    type=float,
    default=0.1,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=256, help="# of images in each batch of data"
)

data_arg.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="# of subprocesses to use for data loading",
)

data_arg.add_argument("--dataset_type", type=str, default="CIFAR-10")

data_arg.add_argument(
    "--cifar_classifier_indexes",
    type=str2intlist,
    default="[0,1,2,3,4,5]",
    help="indexes of the classifiers",
)

data_arg.add_argument(
    "--weak_classifier_folder",
    type=str,
    default="./weak_classifiers/create_classifier_subsets_create_classifier_subsets",
    help="folder with *.npy files",
)
data_arg.add_argument("--n_classes", type=int, default=10, help="n_classes")

# training params
train_arg = add_argument_group("Training Params")

train_arg.add_argument("--optimizer", type=str, default="Adam")

train_arg.add_argument(
    "--epochs", type=int, default=200, help="# of epochs to train for"
)
train_arg.add_argument(
    "--init_lr", type=float, default=3e-4, help="Initial learning rate value"
)
train_arg.add_argument(
    "--lr_milestones",
    type=str2intlist,
    default="[170,190]",
    help="indexes of the classifiers",
)
train_arg.add_argument(
    "--freeze_all_but_stop_net", type=int, default=0, help="freeze all but stop net"
)
train_arg.add_argument(
    "--lr_patience",
    type=int,
    default=20,
    help="Number of epochs to wait before reducing lr",
)


train_arg.add_argument(
    "--skip_validation", type=str2bool, default=False, help="skip valition or not"
)
train_arg.add_argument(
    "--forbid_short_episodes", type=str2bool, default=True, help="forbid_short_episodes"
)

# other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument("--gama", type=float, default=0.98, help="debug_parameter1")


misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=False, help="Whether to run on the GPU"
)

misc_arg.add_argument(
    "--random_seed", type=int, default=1, help="Seed to ensure reproducibility"
)


misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)

misc_arg.add_argument(
    "--load_checkpoint_path",
    type=str,
    default="",
    help="if not None then resume from path",
)

parser.add_argument("--train_set_size", type=int, default=-1)
parser.add_argument("--test_set_size", type=int, default=-1)


def get_config():
    config = smart_parse_args(parser)
    config.output = config.output_dir
    # config = parser.parse_args()
    return config

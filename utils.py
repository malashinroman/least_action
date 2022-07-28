import json
import logging
import os
import shutil

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image


def copy_weight_part(own_state, name, param):
    if own_state[name].shape == param.shape:
        own_state[name].resize_as_(param)
        own_state[name].copy_(param)
    else:
        print(
            f'Skipping weight "{name}" copying. {own_state[name].shape} (dst) !={param.shape} (src)'
        )


def load_state_dict_into_module(state_dict, module, strict=False, ignore_prefix=[]):
    own_state = module.state_dict()
    not_loaded = set(state_dict.keys()) - set(own_state.keys())
    print(
        f"{len(state_dict.keys())}/{len(state_dict.keys()) - len(not_loaded)} weight elements can be loaded"
    )

    for name, param in state_dict.items():
        for pr in ignore_prefix:
            name = name.replace(pr, "")

        all_names_with_prefixes = [name] + [p + name for p in ignore_prefix]
        satisfies = [p in own_state for p in all_names_with_prefixes]
        satisf_indexes = [i for i, x in enumerate(satisfies) if x]
        assert len(satisf_indexes) < 2
        if len(satisf_indexes) > 0:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state_name = all_names_with_prefixes[satisf_indexes[0]]
                copy_weight_part(own_state, own_state_name, param)
            except Exception:
                raise RuntimeError(
                    "While copying the parameter named {}., "
                    "whose dimensions in the model are {} and "
                    "whose dimensions in teh checkpoint are {}".format(
                        name, own_state[name].size(), param.size()
                    )
                )
        elif strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            raise KeyError('unexpected key "{}" in state_dict'.format(missing))
        else:
            print(f"Not loaded: {name}")


def load_checkpoint(model, load_checkpoint_path=None):
    """https://calendar.google.com/calendar/u/0/rhttps://calendar.google.com/calendar/u/0/r
    Load the best copy of a model. This is useful for 2 cases:
    """
    print("[*] Loading model from {}".format(load_checkpoint_path))

    if load_checkpoint_path is not None:
        ckpt_path = load_checkpoint_path
        ckpt = torch.load(ckpt_path)

        load_state_dict_into_module(
            ckpt["model_state"], model, ignore_prefix=["model."]
        )
        epoch = ckpt["epoch"] if "epoch" in ckpt else "unknown"
        valid_acc = "unknown"
        if "valid_acc" in ckpt:
            valid_acc = f"{ckpt['valid_acc']:.3f}"
        elif "best_valid_acc" in ckpt:
            valid_acc = f"{ckpt['best_valid_acc']:.3f}"

        print(
            "[*] Loaded {} checkpoint @ epoch {} \n"
            "with best valid acc of {}".format(ckpt_path, epoch, valid_acc)
        )
    else:
        raise ("provide the checkpoint")


def write_json(data_dict, filename):
    with open(filename, "w") as file:
        json.dump(data_dict, file)


# def denormalize(T, coords):
#     return 0.5 * ((coords + 1.0) * T)


# def bounding_box(x, y, size, color="w"):
#     x = int(x - (size / 2))
#     y = int(y - (size / 2))
#     rect = patches.Rectangle(
#         (x, y), size, size, linewidth=1, edgecolor=color, fill=False
#     )
#     return rect


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def resize_array(x, size):
#     # 3D and 4D tensors allowed only
#     assert x.ndim in [3, 4], "Only 3D and 4D Tensors allowed!"

#     # 4D Tensor
#     if x.ndim == 4:
#         res = []
#         for i in range(x.shape[0]):
#             img = array2img(x[i])
#             img = img.resize((size, size))
#             img = np.asarray(img, dtype="float32")
#             img = np.expand_dims(img, axis=0)
#             img /= 255.0
#             res.append(img)
#         res = np.concatenate(res)
#         res = np.expand_dims(res, axis=1)
#         return res

#     # 3D Tensor
#     img = array2img(x)
#     img = img.resize((size, size))
#     res = np.asarray(img, dtype="float32")
#     res = np.expand_dims(res, axis=0)
#     res /= 255.0
#     return res


# def img2array(data_path, desired_size=None, expand=False, view=False):
#     """
#     Util function for loading RGB image into a numpy array.
#     Returns array of shape (1, H, W, C).
#     """
#     img = Image.open(data_path)
#     img = img.convert("RGB")
#     if desired_size:
#         img = img.resize((desired_size[1], desired_size[0]))
#     if view:
#         img.show()
#     x = np.asarray(img, dtype="float32")
#     if expand:
#         x = np.expand_dims(x, axis=0)
#     x /= 255.0
#     return x


# def array2img(x):
#     """
#     Util function for converting anumpy array to a PIL img.
#     Returns PIL RGB img.
#     """
#     x = np.asarray(x)
#     x = x + max(-np.min(x), 0)
#     x_max = np.max(x)
#     if x_max != 0:
#         x /= x_max
#     x *= 255
#     return Image.fromarray(x.astype("uint8"), "RGB")


# def plot_images(images, gd_truth):
#     images = images.squeeze()
#     assert len(images) == len(gd_truth) == 9
#     # Create figure with sub-plots.
#     fig, axes = plt.subplots(3, 3)
#     for i, ax in enumerate(axes.flat):
#         # plot the image
#         ax.imshow(images[i], cmap="Greys_r")
#         xlabel = "{}".format(gd_truth[i])
#         ax.set_xlabel(xlabel)
#         ax.set_xticks([])
#         ax.set_yticks([])
#     plt.show()


# def prepare_dirs(config):
#     for path in [config.data_dir]:
#         if not os.path.exists(path):
#             os.makedirs(path)


def save_checkpoint(state, is_best, model_name, ckpt_dir):

    """
    Save a copy of the model so that it can be loaded at a future
    date. This function is used when the model is being evaluated
    on the test data.

    If this model has reached the best validation accuracy thus
    far, a seperate file with the suffix `best` is created.
    """
    # print("[*] Saving model to {}".format(self.ckpt_dir))

    filename = model_name + "_latest.pth.tar"
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)

    best_ckpt_path = None
    if is_best:
        filename = model_name + "_best.pth.tar"
        best_ckpt_path = os.path.join(ckpt_dir, filename)
        shutil.copyfile(ckpt_path, best_ckpt_path)
        stats = {
            k: v
            for (k, v) in state.items()
            if type(v) is str or type(v) is int or type(v) is float
        }
        write_json(stats, os.path.join(ckpt_dir, model_name + "_best_val.json"))

    return ckpt_path, best_ckpt_path


def copy_weight_part(own_state, name, param):
    if own_state[name].shape == param.shape:
        own_state[name].resize_as_(param)
        own_state[name].copy_(param)
    else:
        logging.warning(
            f'Skipping weight "{name}" copying. {own_state[name].shape} (dst) !={param.shape} (src)'
        )


def load_state_dict_into_module(state_dict, module, strict=False, ignore_prefix=[]):
    own_state = module.state_dict()
    not_loaded = set(state_dict.keys()) - set(own_state.keys())
    print(
        f"{len(state_dict.keys())}/{len(state_dict.keys()) - len(not_loaded)} weight elements can be loaded"
    )

    for name, param in state_dict.items():
        for pr in ignore_prefix:
            name = name.replace(pr, "")

        all_names_with_prefixes = [name] + [p + name for p in ignore_prefix]
        satisfies = [p in own_state for p in all_names_with_prefixes]
        satisf_indexes = [i for i, x in enumerate(satisfies) if x]
        assert len(satisf_indexes) < 2
        if len(satisf_indexes) > 0:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            try:
                own_state_name = all_names_with_prefixes[satisf_indexes[0]]
                copy_weight_part(own_state, own_state_name, param)
            except Exception:
                raise RuntimeError(
                    "While copying the parameter named {}., "
                    "whose dimensions in the model are {} and "
                    "whose dimensions in teh checkpoint are {}".format(
                        name, own_state[name].size(), param.size()
                    )
                )
        elif strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            raise KeyError('unexpected key "{}" in state_dict'.format(missing))
        else:
            logging.warning(f"Not loaded: {name}")


def save_config(the_config):
    if type(the_config) is wandb.sdk.wandb_config.Config:
        config = {}
        for key, val in the_config.items():
            config[key] = val
    else:
        config = the_config
    model_name = "model"
    filename = model_name + "_params.json"
    if not os.path.exists(config.output):
        os.makedirs(config.output)
    param_path = os.path.join(config.output, filename)

    print("[*] Model Checkpoint Dir: {}".format(config.ckpt_dir))
    print("[*] Param Path: {}".format(param_path))

    with open(param_path, "w") as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def collect_trajectories(net_out, episodes_info, reward_check=False, non_reward=False):
    """Analyzes actions taken by the network and returns
    trajectories of actions
    Args:
        net_out (dict) : output of the network
        episodes_info (dict) : episodes info
        reward_check (bool): if False, then returns all trajectories
        non_reward (bool): if True then returns trajectories with not rewarded actions,
        else rewarded actions (has no effect if reward_check is False)
    """
    trajectories = []
    batch_size = len(net_out["actions_info"]["selected_classifier"][0])
    num_glimpses = len(net_out["actions_info"]["selected_classifier"])
    for k in range(batch_size):
        traj = []
        for j in range(num_glimpses):
            traj.append(net_out["actions_info"]["selected_classifier"][j][k].item())
            if episodes_info["valid_steps"][j][k] <= 0.001:
                break
        if reward_check:
            if episodes_info["steps_reward"][0][k] != 0.0 and not non_reward:
                trajectories.append(traj)
            elif episodes_info["steps_reward"][0][k] == 0.0 and non_reward:
                trajectories.append(traj)
        else:
            trajectories.append(traj)
    return trajectories


def collect_all_trajectories(net_out, episodes_info):
    """Analyzes actions taken by the network and returns"""

    trajectories = []
    batch_size = len(net_out["actions_info"]["selected_classifier"][0])
    num_glimpses = len(net_out["actions_info"]["selected_classifier"])
    for k in range(batch_size):
        traj = []
        for j in range(num_glimpses):
            traj.append(net_out["actions_info"]["selected_classifier"][j][k].item())
            if episodes_info["valid_steps"][j][k] <= 0.001:
                break
        trajectories.append(traj)

    return trajectories


def compute_reward(log_probas, y, nll_reward, clamp_nll_reward=None):
    y = y.to(log_probas.device)
    predicted = torch.max(log_probas, 1)[1]
    if not nll_reward:
        R = (predicted.detach() == y).float()
    else:
        R = -F.nll_loss(log_probas.detach(), y, reduction="none")
        if clamp_nll_reward is not None:
            R = R.clamp(clamp_nll_reward, 0)
    return R


def collect_dynamic_data_dram(model_output, y, gama, config):
    """Collects data from the network
    output taking into account stop action.
    Gets decisions when stop network said to stop.
    Gets mask of valid actions for fast loss computation.
    """
    out_dict = {}

    finished_log_probas = torch.zeros(model_output["all_log_probas"][0].shape).to(
        model_output["baselines"][0].device
    )
    finished_logits = torch.zeros(model_output["all_logits"][0].shape).to(
        model_output["baselines"][0].device
    )
    finished_episodes = torch.zeros(model_output["baselines"][0].shape).to(
        model_output["baselines"][0].device
    )
    episode_length_hist = np.zeros(config.num_glimpses)

    rewards = []
    valid_steps = []
    finish_steps = []
    baseline_traj = []

    for i in range(config.num_glimpses):
        log_probas = model_output["all_log_probas"][i]
        logits = model_output["all_logits"][i]
        if i != config.num_glimpses - 1:
            finished = (model_output["actions_info"]["stop"][i] == 1).float()
            finished = (
                finished
                * (
                    torch.ones(model_output["baselines"][0].shape).to(
                        model_output["baselines"][0].device
                    )
                    - finished_episodes
                ).int()
            )
        else:
            finished = (
                torch.ones(model_output["baselines"][0].shape).to(
                    model_output["baselines"][0].device
                )
                - finished_episodes
            )
            finished = finished.int()
        if gama > 0:
            coeff = np.power(gama, i)
            R = compute_reward(log_probas, y, config.nll_reward) * coeff * finished
        else:
            penalty = gama * i
            R = (compute_reward(log_probas, y, config.nll_reward) + penalty) * finished

        for r in rewards:
            r += R
        for v in valid_steps:
            v += finished
        for b in baseline_traj:
            b += model_output["baselines"][i] * finished
        baseline_traj.append(model_output["baselines"][i] * finished)
        valid_steps.append(finished.clone())
        finish_steps.append(finished.clone())
        assert torch.cat(valid_steps).max() <= 1 and torch.cat(valid_steps).min() >= 0
        assert torch.cat(finish_steps).max() <= 1
        rewards.append(R.clone())
        finished_episodes += finished
        episode_length_hist[i] = finished.sum()
        for j in range(finished.shape[0]):
            if finished[j]:
                finished_log_probas[j] = log_probas[j]
                finished_logits[j] = logits[j]

    out_dict["finished_log_probas"] = finished_log_probas
    out_dict["finished_logits"] = finished_logits
    out_dict["steps_reward"] = rewards
    out_dict["valid_steps"] = valid_steps
    out_dict["finish_steps"] = finish_steps
    out_dict["episode_length_hist"] = episode_length_hist
    out_dict["baseline_traj"] = baseline_traj

    return out_dict

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from plot_glimpses import visualize_trajectories, visualize_trajectories_double
from script_manager.func.wandb_logger import (  # write_wandb_dict,
    write_wandb_bar,
    write_wandb_scalar,
)
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from utils import (
    AverageMeter,
    collect_dynamic_data_dram,
    collect_trajectories,
    save_checkpoint,
)


class ModelOutputTracker:
    """Gathers statistics of the output produced by the network
    that is reported"""

    def __init__(self, config):
        self.config = config
        self.selected_classifiers_hist = np.zeros(
            [self.config.num_glimpses, len(self.config.cifar_classifier_indexes)]
        )

    def update(self, model_output):
        selected = torch.stack(
            model_output["actions_info"]["selected_classifier"], dim=0
        )
        for g in range(self.config.num_glimpses):
            cl_indexes, counts = np.unique(
                selected[g].detach().numpy(), return_counts=True
            )
            for index, count in zip(cl_indexes, counts):
                self.selected_classifiers_hist[g][index] += count

    def reset(self):
        self.selected_classifiers_hist = np.zeros(
            [self.config.num_glimpses, len(self.config.cifar_classifier_indexes)]
        )

    def get_current_info(self):
        return {
            "selected_classifiers_hist": self.selected_classifiers_hist,
            "selected_classifiers_hist_total": self.selected_classifiers_hist.sum(
                axis=0
            ),
        }


class Trainer(object):

    """
    model: enitity converting input to output
    trained_pars: list of modules, which weihgt should update
    data_loader: data_loader
    test_data_loader: test_data_loader
    """

    def __init__(self, config, model, data_loader, test_data_loader, loss_function):
        self.config = config

        # unpack dataloaders
        self.train_loader = data_loader[0]
        self.valid_loader = data_loader[1]
        self.num_train = config.data_size[0]
        self.num_valid = config.data_size[1]

        self.test_loader = test_data_loader
        self.num_test = len(self.test_loader.dataset)

        self.start_epoch = 0

        # initialize paths to save checkpoints
        self.ckpt_dir = os.path.join(self.config.output, config.ckpt_dir)
        self.logs_dir = os.path.join(self.config.output, config.logs_dir)
        self.vals_dir = os.path.join(self.logs_dir, "validation")
        self.train_dir = os.path.join(self.logs_dir, "train")
        self.best_valid_acc = 0.0
        self.best_valid_reward = 0.0
        self.counter = 0
        self.model_name = "model"

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        if not os.path.exists(self.vals_dir):
            os.makedirs(self.vals_dir)
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        self.att_model = model
        print(
            "[*] Number of model parameters: {:,}".format(
                sum([p.data.nelement() for p in self.att_model.parameters()])
            )
        )

        self.optimizer = optim.__dict__[config.optimizer](
            self.att_model.parameters(), lr=config.init_lr
        )

        self.scheduler_base = MultiStepLR(
            self.optimizer, milestones=self.config.lr_milestones
        )

        self.ckpt_best = None
        self.ckpt_latest = None
        self.loss_function = loss_function

    def train(self, start_epoch, epochs):

        # load the most recent checkpoint
        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(start_epoch, start_epoch + epochs):
            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    start_epoch + epoch + 1,
                    start_epoch + epochs,
                    self.optimizer.param_groups[0]["lr"],
                )
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch + 1)

            # evaluate on validation set
            valid_dict = self.validate(epoch + 1)

            self.scheduler_base.step()

            # prepare and write logs
            epoch_info = {
                "train_acc": train_acc,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch": epoch + 1,
            }
            epoch_info.update(valid_dict)

            write_wandb_scalar(
                dict(
                    {"train_epoch_" + key: val for key, val in epoch_info.items()},
                    **{"global_step": epoch + 1},
                ),
                commit=True,
            )

            # console message
            is_best = valid_dict["reward"] > self.best_valid_reward
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - valid_rew {:.3f} "
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss,
                    train_acc,
                    valid_dict["loss"],
                    valid_dict["acc"],
                    valid_dict["reward"],
                )
            )

            self.best_valid_acc = max(valid_dict["acc"], self.best_valid_acc)
            self.best_valid_reward = max(valid_dict["reward"], self.best_valid_reward)

            chkpt_wiht_meta_info = {
                "epoch": epoch + 1,
                "model_state": self.att_model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
            }
            chkpt_wiht_meta_info.update(valid_dict)
            self.ckpt_latest, ckpt_best = save_checkpoint(
                chkpt_wiht_meta_info, is_best, self.model_name, self.ckpt_dir
            )
            if is_best:
                self.ckpt_best = ckpt_best
                # self.test(skip_load=True, self.)

        return self.best_valid_acc

    def update_total_dict(self, loss_total_dict, loss_dict):
        for key, value in loss_dict.items():
            if key not in loss_total_dict:
                if type(value) == torch.Tensor:
                    loss_total_dict[key] = value.item()
                else:
                    loss_total_dict[key] = value
            else:
                if type(value) == torch.Tensor:
                    loss_total_dict[key] += value.item()
                else:
                    loss_total_dict[key] += value

    def train_one_epoch(self, epoch):
        self.att_model.train()
        output_tracker = ModelOutputTracker(self.config)
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        tic = time.time()
        trajectories = []
        trajectories_rewarded = []
        histograms = []
        for _ in range(self.config.num_glimpses):
            histograms.append([])

        loss_total_dict = {}

        with tqdm(total=self.num_train) as pbar:
            for _, data in enumerate(self.train_loader):

                x = data["image"]
                y = data["label"]
                if self.config.use_gpu:
                    x, y = x.cuda(), y.cuda()

                self.batch_size = x.shape[0]
                dict_log = {}

                self.optimizer.zero_grad()
                self.att_model.reset(self.batch_size)

                # run model for num_glimpses steps
                model_output = self.att_model.forward(data, self.config.num_glimpses)
                output_tracker.update(model_output=model_output)

                # collect info for loss calculation
                episodes_info = collect_dynamic_data_dram(
                    model_output, y, self.config.gama, self.config
                )

                loss_dict = self.loss_function(model_output, episodes_info, data)
                loss_dict["loss"].backward()
                # output_dict =
                # output_dict.update(),
                write_wandb_scalar(
                    dict(
                        {"iter_" + k: v for k, v in loss_dict.items()},
                        **{"global_step": epoch + 1},
                    ),
                    commit=False,
                )

                self.optimizer.step()
                dict_log.update(loss_dict)

                accs.update(loss_dict["acc"].item(), len(y))
                losses.update(loss_dict["loss"].item(), len(y))
                self.update_total_dict(loss_total_dict, loss_dict)

                # collect info to visualize trajectories
                trajectories += collect_trajectories(
                    model_output, episodes_info, reward_check=True, non_reward=True
                )
                trajectories_rewarded += collect_trajectories(
                    model_output, episodes_info, reward_check=True
                )

                # to measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)
                if type(self.scheduler_base) == torch.optim.lr_scheduler.OneCycleLR:
                    self.scheduler_base.step()

                pbar.set_description(
                    (
                        f"adj_rew: {dict_log['adjusted_reward']:.3f} "
                        f"reward {dict_log['reward'].item():.3f} "
                        f"train loss {dict_log['loss'].item():.3f} "
                        f"entropy (var/unc) {dict_log['entropy_var'].item():.3f} / {dict_log['entropy_unc']}"
                    )
                )

                pbar.update(self.batch_size)

        visualize_trajectories_double(
            trajectories_rewarded,
            trajectories,
            [repr(len(self.config.cifar_classifier_indexes))],
            os.path.join(self.train_dir, "visualized_policy{:07d}.dot".format(epoch)),
        )

        write_wandb_scalar(
            dict(
                {
                    "epoch_summary_" + key: val / len(self.train_loader)
                    for key, val in loss_total_dict.items()
                },
                **{"global_step": epoch + 1},
            ),
            commit=False,
        )

        info = output_tracker.get_current_info()
        write_wandb_bar(
            "train_classifier_distribution",
            info["selected_classifiers_hist_total"],
            indexes_label="classifiers",
            height_label="calls",
            commit=False,
        )

        return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        self.att_model.model.eval()
        output_tracker = ModelOutputTracker(self.config)
        losses = AverageMeter()
        accs = AverageMeter()
        rews = AverageMeter()

        histograms = []
        ep_lenght_hist = []
        trajectories = []
        trajectories_rewarded = []
        correct_num = 0
        total_num = 0
        predicted_actions_all = []

        for i, data in enumerate(self.valid_loader):
            x = data["image"]
            y = data["label"]
            if self.config.use_gpu:
                x, y = x.cuda(), y.cuda()

            self.batch_size = x.shape[0]
            self.att_model.reset(self.batch_size)

            with torch.no_grad():
                net_out = self.att_model.forward(data, self.config.num_glimpses)

            output_tracker.update(model_output=net_out)

            episodes_info = collect_dynamic_data_dram(
                net_out, y, self.config.gama, self.config
            )
            trajectories += collect_trajectories(
                net_out, episodes_info, reward_check=True, non_reward=True
            )
            trajectories_rewarded += collect_trajectories(
                net_out, episodes_info, reward_check=True
            )

            sel = [
                i.cpu().detach().numpy()
                for i in net_out["actions_info"]["selected_classifier"]
            ]
            selected = []

            for k in range(len(sel)):
                for p in range(len(sel[k])):
                    if episodes_info["valid_steps"][k][p] > 0:
                        selected.append(sel[k][p])

            hists = np.array(selected)

            classifiers = hists.reshape(-1)

            hist = np.histogram(
                classifiers,
                bins=len(self.config.cifar_classifier_indexes),
                range=(-0.5, len(self.config.cifar_classifier_indexes) - 0.5),
            )[0]
            histograms.append(hist)

            ep_lenght_hist.append(episodes_info["episode_length_hist"])

            lp = net_out["all_log_probas"][-1]
            predicted = torch.max(lp, 1)[1]
            rewards = ((predicted == y).float()).mean()

            correct = (predicted == y).float()
            correct_num += correct.sum().item()
            total_num += y.shape[0]
            acc = 100 * (correct.sum() / len(y))
            accs.update(acc.item(), x.size()[0])

            rews.update(rewards.item(), x.size()[0])

            predicted_actions = [
                (
                    net_out["oracle_logprob"][c].argmax(dim=1)
                    == net_out["actions_info"]["selected_classifier"][c]
                )
                .float()
                .sum()
                for c in range(len(net_out["oracle_logprob"]))
            ]
            predicted_actions_all.append(predicted_actions)
        # log to dot-file
        visualize_trajectories_double(
            trajectories_rewarded,
            trajectories,
            [repr(len(self.config.cifar_classifier_indexes))],
            os.path.join(self.vals_dir, "visualized_policy{:07d}.dot".format(epoch)),
        )
        # log to tensorboard
        print(f"correct: {correct_num} / {total_num}")
        print(
            f"predicted action %: {np.array(predicted_actions_all).sum(axis=0) / self.num_valid * 100}"
        )

        overal_hist = np.array(histograms).sum(0)
        episodes_length = np.array(ep_lenght_hist).sum(0)
        write_wandb_bar(
            tag="val_claissfier_distribution_tmp",
            bars_val=overal_hist,
            indexes_label="classifier index",
            height_label="calls",
            commit=False,
        )

        output_info = output_tracker.get_current_info()
        write_wandb_bar(
            tag="val_claissfier_distribution",
            bars_val=output_info["selected_classifiers_hist_total"],
            indexes_label="classifier index",
            height_label="calls",
            commit=False,
        )
        for i in range(len(range(self.config.num_glimpses))):
            write_wandb_bar(
                tag=f"val_claissfier_distribution_step{i}",
                bars_val=output_info["selected_classifiers_hist"][i],
                indexes_label="classifier index",
                height_label="calls",
                commit=False,
            )

        plt.close()

        average_ep_length = (
            np.array(
                [(i + 1) * episodes_length[i] for i in range(len(episodes_length))]
            ).sum()
            / episodes_length.sum()
        )

        return {
            "loss": losses.avg,
            "acc": accs.avg,
            "reward": rews.avg,
            "average_ep_length": average_ep_length,
        }

    def test(self, skip_load=False, data_loader=None, file_prefix="u_"):
        """
        Additional test function. This function is called after the training is finished.
        I don't remember why I didn't reuse validation function.
        """

        print(f"{file_prefix}testing")
        correct = 0

        # load the best checkpoint
        if not skip_load:
            self.load_checkpoint(
                best=False, load_checkpoint_path=self.config.load_checkpoint_path
            )
        self.att_model.eval()

        trajectories = []
        trajectories_rewarded = []
        correct_num = 0
        total_num = 0
        responses = torch.zeros([len(data_loader.dataset), self.config.n_classes])
        responses[...] = -10000
        ep_lenght_hist = []
        for i, data in enumerate(data_loader):
            x = data["image"]
            y = data["label"]
            x = x.to(self.att_model.device)
            y = y.to(self.att_model.device)
            self.batch_size = x.shape[0]
            self.att_model.reset(self.batch_size)
            with torch.no_grad():
                net_out = self.att_model.forward(data, self.config.num_glimpses)

            episodes_info = collect_dynamic_data_dram(
                net_out, y, self.config.gama, self.config
            )
            ep_lenght_hist.append(episodes_info["episode_length_hist"])

            trajectories += collect_trajectories(
                net_out, episodes_info, reward_check=True, non_reward=True
            )
            trajectories_rewarded += collect_trajectories(
                net_out, episodes_info, reward_check=True
            )

            pred = episodes_info["finished_log_probas"].data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            correct_n = (pred[:, 0] == y).float()
            correct_num += correct_n.sum().item()
            responses[
                i * self.config.batch_size : (i + 1) * self.config.batch_size, :
            ] = episodes_info["finished_log_probas"]

            total_num += y.shape[0]

        print(f"correct: {correct_num} / {total_num}")
        output_dot_file = file_prefix + "visualized_policy.dot"
        visualize_trajectories_double(
            trajectories_rewarded,
            trajectories,
            [repr(len(self.config.cifar_classifier_indexes))],
            os.path.join(self.config.output, output_dot_file),
        )
        visualize_trajectories(
            trajectories + trajectories_rewarded,
            terminal_nodes=[repr(i) for i in self.config.cifar_classifier_indexes],
            output=os.path.join(
                self.config.output, file_prefix + "visualized_policy_simple.dot"
            ),
        )
        perc = (100.0 * correct) / (len(data_loader.dataset))
        error = 100 - perc

        print(
            f"[*] {file_prefix} acc: {correct}/{len(data_loader.dataset)} ({perc:.2f}% - {error:.2f}%)"
        )

        episodes_length = np.array(ep_lenght_hist).sum(0)
        average_ep_length = (
            np.array(
                [(i + 1) * episodes_length[i] for i in range(len(episodes_length))]
            ).sum()
            / episodes_length.sum()
        )
        return {"acc": perc.item(), "average_ep_length": average_ep_length}

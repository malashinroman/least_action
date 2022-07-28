import torch
import torch.nn.functional as F


class SparseEnsembleLoss(torch.nn.Module):
    def compute_reward(self, log_probas, y):
        predicted = torch.max(log_probas, 1)[1]
        if not self.config.nll_reward:
            R = (predicted.detach() == y).float()
        else:
            R = -F.nll_loss(log_probas.detach(), y, reduction="none")
            if self.config.clamp_nll_reward is not None:
                R = R.clamp(self.config.clamp_nll_reward, 0)
        return R

    def __init__(self, config):
        super(SparseEnsembleLoss, self).__init__()
        self.config = config

    def forward(self, model_output, episodes_info, data):
        x = data["image"]
        y = data["label"]
        if self.config.use_gpu:
            x, y = x.cuda(), y.cuda()

        R = torch.stack(episodes_info["steps_reward"]).transpose(1, 0)
        valid_steps = torch.stack(episodes_info["valid_steps"]).transpose(1, 0)
        valid_steps_num = valid_steps.sum()
        finish_steps = torch.stack(episodes_info["finish_steps"]).transpose(1, 0)
        finish_steps_num = finish_steps.sum()

        log_probas = episodes_info["finished_log_probas"]
        predicted = torch.max(episodes_info["finished_log_probas"], 1)[1]
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        # 1. supervised loss
        if not self.config.intermediate_supervision:
            supervised_loss = F.nll_loss(log_probas, y)
        else:
            supervised_loss = torch.Tensor([0]).to(log_probas.device)
            for lp in model_output["all_log_probas"]:
                supervised_loss += F.nll_loss(lp, y)
            supervised_loss /= len(model_output["all_log_probas"])

        # 2. baseline loss
        baselines = torch.stack(model_output["baselines"]).transpose(1, 0)
        if self.config.baseline_loss == "mse":
            if self.config.long_shot_baseline:
                loss_baseline = (
                    (baselines - R) * (baselines - R) * valid_steps
                ).sum() / valid_steps_num
            else:
                loss_baseline = (
                    (baselines - R) * (baselines - R) * finish_steps
                ).sum() / finish_steps_num
        elif self.config.baseline_loss == "l1":
            loss_baseline = F.l1_loss(baselines, R, reduction="mean") * 100
        else:
            raise (
                "unknown self._args.baseline_loss {}".format(self._args.baseline_loss)
            )

        # relative reward
        if self.config.long_shot_baseline:
            # FIXME: do we need to detach()?
            bl_traj = baselines
        else:
            bl_traj = (
                torch.stack(episodes_info["baseline_traj"]).transpose(1, 0).detach()
            )

        adjusted_reward = (R - bl_traj) * valid_steps

        # 4. selection loss
        log_pi_action = torch.stack(
            model_output["actions_info"]["classifier_log_prob"]
        ).transpose(1, 0)

        # 4.1 easiest part is reinforce part
        loss_actions = torch.sum(-log_pi_action * adjusted_reward, dim=1).mean(dim=0)

        # 4.2 exploration bonuses
        # use whole distribution to support exploration
        a_logits = model_output["actions_info"]["actions_log_prob"]
        a_logits_indep = torch.cat(a_logits, dim=0)

        # average (across different examples) prob distribution
        # of selecting a classifier on each step
        classifier_probs = []
        for i in range(len(a_logits)):
            classifier_probs.append(torch.exp(a_logits[i]).mean(0))

        # 4.2.1 Measure distributions "variance": individually predicted distributions of actions
        # relative to average prob distribution
        context_significance = []
        # we ignore first choice. It has no context knowledge
        # and should be based on a prior knowledge
        for i in range(1, len(a_logits)):
            context_significance.append(
                (torch.exp(a_logits[i]) - classifier_probs[i]).pow(2).mean()
            )

        if len(context_significance) > 0:
            context_significance = torch.stack(context_significance).mean()
            context_loss = self.config.context_sig_coeff * context_significance
        else:
            context_loss = torch.Tensor([0]).to(a_logits_indep.device)

        # entropy exists for actions >= 2)
        if len(classifier_probs) > 1:
            # 4.2.2 entropy bonus for variativity of actions
            classifier_probs = torch.stack(classifier_probs, dim=1)
            classifier_probs = classifier_probs.transpose(0, 1)[1:]
            entropy_var = (
                -self.config.entropy_var_coeff
                * (classifier_probs * torch.log(classifier_probs)).mean()
            )

            # 4.2.3 entropy bonus to smooth the delta-shaped distributions
            # (action net is encouraged to predict several alternative actions,
            # which we'll support exploration)
            prob_alternative = F.softmax(a_logits_indep, dim=1)
            entropy_unc = (
                -self.config.entropy_unc_coeff
                * (a_logits_indep * prob_alternative).sum(dim=1).mean()
            )

            # final entropy bonus and reinforcement loss
            full_enropy_bonus = -self.config.action_entropy_coeff * (
                entropy_var + entropy_unc + context_loss
            )
        else:
            full_enropy_bonus = torch.Tensor([0]).to(a_logits_indep.device)
            entropy_var = torch.Tensor([0]).to(a_logits_indep.device)
            entropy_unc = torch.Tensor([0]).to(a_logits_indep.device)

        reinforcment_act = loss_actions + full_enropy_bonus

        # 5.0 stop network loss
        stop_actions_log_prob = model_output["actions_info"]["stop_actions_log_prob"]
        stop_regularizer = torch.Tensor([0]).to(stop_actions_log_prob[0].device)
        for i in range(1, len(stop_actions_log_prob)):
            alp = stop_actions_log_prob[i]
            alps = F.softmax(alp, dim=1)
            alps_batch = alps.mean(dim=0)
            stop_regularizer += -torch.sum(alps_batch * alps_batch.log())
        entropy_stop = stop_regularizer / (len(stop_actions_log_prob) + 1)
        entropy_stop_bonus = -self.config.stop_exploration_reg * entropy_stop

        stop_log_prob = (
            torch.stack(model_output["actions_info"]["stop_log_prob"])
            .transpose(1, 0)
            .squeeze(-1)
        )
        loss_stop = (
            torch.sum(-stop_log_prob * adjusted_reward, dim=1).mean(dim=0)
            + entropy_stop_bonus
        )

        # full loss
        loss = (
            self.config.baseline_loss_coeff * loss_baseline
            + self.config.supervised_loss_coeff * supervised_loss
            + self.config.reinforce_act_coeff * reinforcment_act
            + self.config.reinforce_stop_coeff * loss_stop
        )

        # + self.config.reinforce_loc_coeff * loss_reinforce_loc

        return {
            "loss": loss,
            "acc": acc,
            "context_loss": context_loss,
            "entropy_var": entropy_var,
            "entropy_unc": entropy_unc,
            "entropy_loss_v": full_enropy_bonus,
            "reinforcment_act": reinforcment_act,
            "full_loss": loss,
            "adjusted_reward": adjusted_reward.mean(),
            "reward": episodes_info["steps_reward"][0].mean(),
            # "stop_loss": loss_stop,
        }

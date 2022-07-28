"""Core components of Least Action Classifier"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from torch.distributions import Normal


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


class encoder_return_resp_sm(nn.Module):
    """Action encoder for the LAC model."""

    def __init__(self, config):
        self.classifiers_num = len(config.cifar_classifier_indexes)
        self.config = config
        super(encoder_return_resp_sm, self).__init__()

    def forward(self, response, action_info):
        selected = action_info["selected_classifier"]
        code_dict = {}

        if self.config.use_gpu:
            code_dict["selected"] = selected.cuda()
            code_dict["response"] = response.cuda()
        else:
            code_dict["selected"] = selected
            code_dict["response"] = response
        return code_dict


class baseline_network(nn.Module):
    def __init__(self, config, input_size, output_size):
        super(baseline_network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_size),
        )
        self.config = config

    def forward(self, h_t):
        b_t = self.fc(h_t.detach())
        return b_t


class action_generator(nn.Module):
    """action generator maps state to the index of the classifeir"""

    def train(self, mode=None):
        self.evaluation = not mode

    def __init__(self, config, input_size, _, action_prob_size):
        super(action_generator, self).__init__()
        self.config = config
        self.std = torch.nn.Parameter(torch.Tensor([0.05]), requires_grad=True)

        hid_size = self.config.action_net_hidden_size
        self.fc = nn.Linear(input_size, hid_size)

        self.fc_logprob1 = nn.Sequential(nn.Linear(hid_size, action_prob_size))

    def forward(self, h_t):
        feat = F.relu(self.fc(h_t.detach()))

        # action network
        action_logits = self.fc_logprob1(feat)
        actions_log_prob = F.log_softmax(action_logits, dim=1)

        # during evaluation, we use action with maximum estimated probability
        # during training we sample from the predicted distributions
        if not self.evaluation:
            m = torch.distributions.categorical.Categorical(
                logits=action_logits.detach()
            )
            selected_classifier = m.sample()
        else:
            selected_classifier = action_logits.argmax(dim=1)

        classifier_log_prob = torch.gather(
            actions_log_prob, 1, selected_classifier.unsqueeze(1)
        )

        out_dict = {
            "selected_classifier": selected_classifier,
            "classifier_log_prob": classifier_log_prob.squeeze(),
            "actions_log_prob": actions_log_prob,
        }

        return out_dict


class stop_network(nn.Module):
    """stop network is described in
    the opt journal paper,
    can be used to stop compuations conditioned on the system state
    (not used now to be added later)"""

    def __init__(self, config, input_size):
        super(stop_network, self).__init__()
        self.config = config
        self.extra_hidden_size = 64

        hid_size = input_size // 2
        self.fc3 = nn.Linear(input_size + self.extra_hidden_size, hid_size)

        self.fc_stopprob2 = nn.Linear(hid_size, 2)
        self.apply(init_weights)
        pass

    def train(self, mode=None):
        self.evaluation = not mode

    def forward(self, h_t, h_t2):
        h_t3 = torch.cat([h_t.detach(), h_t2], dim=1)
        feat2 = F.relu(self.fc3(h_t3))

        # stop network
        stop_output = self.fc_stopprob2(feat2)
        stop_actions_log_prob = torch.log_softmax(stop_output, dim=1)

        if not self.evaluation:
            m2 = torch.distributions.Categorical(logits=stop_actions_log_prob.detach())
            stop = m2.sample()
        else:
            stop = stop_actions_log_prob.argmax(dim=1)

        stop_log_prob = torch.gather(stop_actions_log_prob, 1, stop.unsqueeze(1))

        out_dict = {
            "stop_actions_log_prob": stop_actions_log_prob,
            "stop_log_prob": stop_log_prob,
            "stop": stop,
        }

        return out_dict


class state_refresher_sm(nn.Module):
    """Stores table with classfier responses and updates it with new responses"""

    def __init__(self, config=None):
        super(state_refresher_sm, self).__init__()
        self.config = config
        self.batch_size = 128
        self.classifier_num = len(config.cifar_classifier_indexes)
        self.table_cols = self.classifier_num
        # initialized with zeros
        self.responses = torch.zeros(
            (self.batch_size, self.table_cols, config.n_classes), requires_grad=False
        )
        self.mask = torch.zeros((self.batch_size, self.table_cols, config.n_classes))
        self.state_info = {}
        self.state_info["h_t"] = self.get_state()

        self.reset(self.batch_size)

    def forward(self, env_act_code):
        selected = env_act_code["selected"]
        response = env_act_code["response"]
        # update table
        self.responses[(list(range(self.batch_size)), selected.tolist())] = response
        self.mask[(list(range(self.batch_size)), selected.tolist())] = 1
        # immediately return state
        self.state_info["h_t"] = self.get_state()
        return

    def get_state(self):
        if self.config.use_mask_state:
            h_t = torch.cat(
                [
                    self.responses.view(self.batch_size, -1),
                    self.mask.view(self.batch_size, -1),
                ],
                dim=1,
            )
        else:
            h_t = torch.cat([self.responses.view(self.batch_size, -1)])

        self.state_info["h_t"] = h_t

        return h_t

    def reset(self, batch_size):
        """refills the table with zeros, allocates memory for a batch of images"""

        dtype = torch.cuda.FloatTensor if self.config.use_gpu else torch.FloatTensor
        device = "cuda:0" if self.config.use_gpu else "cpu:0"
        self.batch_size = batch_size
        self.responses = torch.zeros(
            (self.batch_size, self.table_cols, self.config.n_classes)
        ).to(device)
        self.mask = torch.zeros(
            (self.batch_size, self.table_cols, self.config.n_classes)
        ).to(device)

        # h_t2 is for stop network only
        self.state_info["h_t"] = self.get_state()
        h_t2 = torch.zeros(batch_size, 64)
        h_t2 = Variable(h_t2).type(dtype)
        self.state_info["h_t2"] = h_t2

    def get_state_size(self):
        """vectorizes the table"""
        if self.config.use_mask_state:
            return (
                self.responses.shape[1] * self.responses.shape[2]
                + self.mask.shape[1] * self.mask.shape[2]
            )
        else:
            return self.responses.shape[1] * self.responses.shape[2]


class descision_maker_deep(nn.Module):
    """subnetwork that is responsible for producing lables"""

    def __init__(self, input_size, output_size):
        super(descision_maker_deep, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, h_t):
        """input : h_t - hidden state,
        output    : log_prob of actions, logits of the network output
        """
        logits = self.fc(h_t)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob, logits


class LacModel(nn.Module):
    """LAC model"""

    def __init__(self, config):
        super(LacModel, self).__init__()

        self.config = config
        num_classes = config.n_classes

        self.resp_action_encoder = encoder_return_resp_sm(config)
        action_prob_size = len(self.config.cifar_classifier_indexes)
        self.state_controller = state_refresher_sm(config)
        hidden_size = self.state_controller.get_state_size()
        self.classifier = descision_maker_deep(hidden_size, num_classes)

        self.actioner = action_generator(
            config, hidden_size, 2, action_prob_size=action_prob_size
        )

        self.baseliner = baseline_network(config, hidden_size, 1)
        self.stop_net = stop_network(config, hidden_size)

    def forward(self, data):

        # first use previous state to select an action
        action_info = self.actioner(self.state_controller.state_info["h_t"])

        # baseline estimation of the reward without taking into account the action
        b_t = self.baseliner(self.state_controller.state_info["h_t"]).squeeze()

        action = action_info["selected_classifier"]

        # collect responses according to actions taken
        env_response = torch.gather(
            data["cifar_env_response"],
            1,
            action.view(-1, 1, 1).repeat(1, 1, 10),
        ).squeeze()

        # encode the response and update the state
        env_act_code = self.resp_action_encoder(env_response, action_info)
        self.state_info = self.state_controller(env_act_code)

        # make decision accoring to new state
        log_probas, logits = self.classifier(self.state_controller.state_info["h_t"])

        out_dict = {"b_t": b_t, "action_info": action_info}
        out_dict["log_probas"] = log_probas
        out_dict["logits"] = logits

        # stop network has no effect at the moment
        stop_info = self.stop_net(
            self.state_controller.state_info["h_t"],
            self.state_controller.state_info["h_t2"],
        )

        for key, val in stop_info.items():
            action_info[key] = val

        return out_dict

    def reset(self, batch_size):
        self.state_controller.reset(batch_size)

    def get_state(self):
        return self.state_controller.get_state()

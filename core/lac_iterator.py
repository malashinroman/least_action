import torch


class LacIterator(torch.nn.Module):
    """Wrapper around lac_core
    to iterate fixed number of times
    """

    def __init__(self, config, la_model):
        super(LacIterator, self).__init__()
        self.config = config
        self.model = la_model
        self.device = "cpu"
        if self.config.use_gpu:
            self.device = "cuda:0"

    def list_of_dict2dict_of_lists(self, list):
        v = {k: [dic[k] for dic in list] for k in list[0]}
        return v

    # def set_env(self, is_train):
    #     self.model.env.set_dataset(is_train)

    def append_out_dict(self, out_dict, net_out):
        for key, val in net_out.keys():
            out_dict[key].append(val)

    def forward(self, data, num_glimpses):
        out_dict = {"oracle_logprob": [], "hidden_states": []}
        legacy_names_mapping = {
            "b_t": "baselines",
            "log_probas": "all_log_probas",
            "logits": "all_logits",
        }

        self.model.reset(data["label"].shape[0])

        for _ in range(num_glimpses):
            out_dict["hidden_states"].append(self.model.get_state())
            net_out = self.model(data)

            if self.config.forbid_short_episodes:
                net_out["action_info"]["stop"] = torch.zeros(
                    net_out["b_t"].shape, requires_grad=False
                ).to(self.device)

            for k, v in net_out.items():
                if k in out_dict:
                    out_dict[k].append(v)
                else:
                    out_dict[k] = [v]

        out_dict["actions_info"] = self.list_of_dict2dict_of_lists(
            out_dict["action_info"]
        )

        # should be cleaned up in the future
        for or_key, upd_key in legacy_names_mapping.items():
            if or_key in out_dict:
                out_dict[upd_key] = out_dict[or_key]

        return out_dict

    def reset(self, batch_size):
        self.model.reset(batch_size)

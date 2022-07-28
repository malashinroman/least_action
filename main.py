import os

import torch
import wandb
from config import get_config
from core.lac_core import LacModel
from core.lac_iterator import LacIterator
from core.sparse_ensemble_loss import SparseEnsembleLoss
from core.trainer import Trainer
from data_loader import get_test_loader, get_train_valid_loader2
from script_manager.func.wandb_logger import write_wandb_scalar
from utils import load_checkpoint, write_json


def verify_model(trainer, train_dataset, val_dataset, test_dataset, out_json):
    """Compute the accuracy of the model on the train, val and test sets."""
    test_dict = trainer.test(
        skip_load=True,
        data_loader=torch.utils.data.DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        ),
        file_prefix="(dataset=test set)",
    )

    train_dict = trainer.test(
        skip_load=True,
        data_loader=torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        ),
        file_prefix="(dataset=train set)",
    )

    val_dict = trainer.test(
        skip_load=True,
        data_loader=torch.utils.data.DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
        ),
        file_prefix="(dataset=val set)",
    )

    train_dict = {f"train_{k}": v for (k, v) in train_dict.items()}
    val_dict = {f"val_{k}": v for (k, v) in val_dict.items()}
    test_dict = {f"test_{k}": v for (k, v) in test_dict.items()}
    dict = {}
    dict.update(train_dict)
    dict.update(val_dict)
    dict.update(test_dict)

    write_json(dict, out_json)
    return dict


def main(config):
    # ensure directories are setup
    # prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)

    kwargs = {"num_workers": config.num_workers}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 0, "pin_memory": True}

    # instantiate data loaders first
    data_loader, data_size, _ = get_train_valid_loader2(
        args=config,
        dataset_type=config.dataset_type,
        batch_size=config.batch_size,
        valid_size=config.valid_size,
        kwargs=kwargs,
    )

    config.data_size = data_size

    test_data_loader, _, _ = get_test_loader(
        config,
        dataset_type=config.dataset_type,
        batch_size=config.batch_size,
        kwargs=kwargs,
    )

    # instantiate the model
    model = LacModel(config)
    if config.use_gpu:
        model.cuda()
    att_model = LacIterator(config, model)
    if config.use_gpu:
        att_model.cuda()

    print(config)
    loss_function1 = SparseEnsembleLoss(config)

    trainer_complete = Trainer(
        model=att_model,
        data_loader=data_loader,
        test_data_loader=test_data_loader,
        config=config,
        loss_function=loss_function1,
    )

    # train from for the specified number of epochs
    trainer_complete.train(0, config.epochs)

    log_latest_json = os.path.join(trainer_complete.ckpt_dir, "model_latest_stats.json")
    log_best_json = os.path.join(trainer_complete.ckpt_dir, "model_best_stats.json")

    result_dict = verify_model(
        trainer_complete,
        data_loader[0].dataset,
        data_loader[1].dataset,
        test_data_loader.dataset,
        log_latest_json,
    )

    # if config.wandb_project_name is not None:
    write_wandb_scalar({"latest_" + key: val for key, val in result_dict.items()})

    # we can have 0 epochs of training
    if trainer_complete.ckpt_best is not None:
        load_checkpoint(trainer_complete.att_model, trainer_complete.ckpt_best)
        result_dict = verify_model(
            trainer_complete,
            data_loader[0].dataset,
            data_loader[1].dataset,
            test_data_loader.dataset,
            log_best_json,
        )
        # if trainer_complete.writer is not None:
        #     trainer_complete.writer.add_scalars("best", result_dict)

        write_wandb_scalar({"best_" + key: val for key, val in result_dict.items()})

    pass


if __name__ == "__main__":
    config = get_config()
    main(config)
    wandb.finish()

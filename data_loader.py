from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data

import torchvision
from torchvision import transforms

def list_files_in_folder(folder, pattern="*test_responses.npy"):
    all_nets = list(Path(folder).rglob(pattern))
    all_nets = [str(n) for n in all_nets]
    return sorted(all_nets)


def get_cifar_env_response_files2(classifiers_indexes, weak_classifier_folder):
    """Returns a list of files with responses for each classifier.
    The files are sorted by classifier index.

    Args:
    classifiers_indexes    : list of classifier indexes to load
    weak_classifier_folder : folder with classifier responses
    Returns                : list of files with responses for each classifier
    """

    classifier_responses = []

    test_resp_files = list_files_in_folder(
        weak_classifier_folder, "*test_repsonses*npy"
    )
    train_resp_files = list_files_in_folder(
        weak_classifier_folder, "*train_repsonses*npy"
    )
    if len(classifiers_indexes) == 0:
        classifiers_indexes = list(range(len(test_resp_files)))

    test_resp_files_used = []
    train_resp_files_used = []
    for i in classifiers_indexes:
        merged_resp = np.concatenate(
            (np.load(train_resp_files[i]), np.load(test_resp_files[i])), axis=0
        )
        test_resp_files_used.append(test_resp_files[i])
        train_resp_files_used.append(train_resp_files[i])
        classifier_responses.append(merged_resp)

    return classifier_responses


def get_cifar_env_response_files(config):
    """Wrapper to simplify access to test set"""

    return get_cifar_env_response_files2(
        config.cifar_classifier_indexes,
        config.weak_classifier_folder,
    )


class IndexedDataset(data.Dataset):
    """Wrapper around torchvision dataset to add indexing,
    and to add cifar_env_response.
    Train and test images are used jointly:
    this gives every image its unique index.
    """

    def __init__(self, args, cifar, index_correction=0):
        self.cifar = cifar
        self._args = args
        self.size = len(self.cifar)
        self.replicates = 1
        # index correction is used to distinguish train and test samples
        self.index_correction = index_correction
        # find npy files with classifier responses
        # and stack train and test responses
        all_responses = get_cifar_env_response_files(args)
        self.all_responses = np.stack(all_responses, axis=1)

    def __getitem__(self, index):
        image, label = self.cifar[index]
        corrected_index = index + self.index_correction
        out_dict = {"image": image, "label": label, "index": corrected_index}

        resp = self.all_responses[corrected_index]
        out_dict["cifar_env_response"] = resp

        return out_dict

    def __len__(self):
        return self.size * self.replicates


def get_train_valid_loader2(args, dataset_type, batch_size, valid_size=0.1, kwargs={}):
    """Prepare data loaders for training and validation.
    uses torchvision cifar dataset,
    but appends every image with its classifier response
    """

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert valid_size <= 1, error_msg

    if dataset_type == "CIFAR-10" or dataset_type == "CIFAR-100":
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
                ),
            ]
        )

        if dataset_type == "CIFAR-10":
            dataset_train = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=trans
            )
            dataset_test = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=trans
            )
            args.n_classes = 10

        elif dataset_type == "CIFAR-100":
            dataset_train = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True, transform=trans
            )
            dataset_test = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=trans
            )

            args.n_classes = 100

        # args.dataset_color = True
        # args.num_channels = 3
        # args.dataset_image_shape = (3, 32, 32)

    dataset_train = IndexedDataset(args, dataset_train)
    dataset_test = IndexedDataset(args, dataset_test, len(dataset_train))

    if valid_size > 0:
        num_train = len(dataset_train)
        num_val = int(valid_size * num_train)
        num_train = num_train - num_val

        dataset_train, dataset_val = torch.utils.data.random_split(
            dataset_train,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(100),
        )
    else:
        dataset_val = dataset_test

    indices_train = list(range(len(dataset_train)))
    indices_test = list(range(len(dataset_train)))

    if args.train_set_size > 0:
        indices_train = indices_train[0 : args.train_set_size]

    if args.test_set_size > 0:
        indices_test = indices_test[0 : args.test_set_size]

    dataset_train = torch.utils.data.Subset(dataset_train, indices_train)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, **kwargs
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, **kwargs
    )

    return (
        (train_loader, valid_loader),
        (len(dataset_train), len(dataset_val)),
        (dataset_train, dataset_val),
    )


def get_test_loader(args, dataset_type, batch_size, kwargs={}):
    res = get_train_valid_loader2(
        args, dataset_type, batch_size, valid_size=-1, kwargs=kwargs
    )
    return res[0][1], res[1][1], res[2][1]

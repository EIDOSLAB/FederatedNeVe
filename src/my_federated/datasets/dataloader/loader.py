import os
import shutil

import PIL
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from my_federated.datasets.custom import (get_mnist, get_cifar, get_caltech,
                                          get_eurosat, get_imagenette, get_medmnist, RandomDataset)
from my_federated.datasets.custom import (get_mnist_transforms, get_cifar_transforms, get_caltech_transforms,
                                          get_eurosat_transforms, get_imagenette_transforms, get_medmnist_transforms)
from my_federated.datasets.dataloader.flwr_fedavgm_common import create_lda_partitions
from my_federated.utils.metrics import get_label_distribution


class NumpyDataset(Dataset):
    def __init__(self, x, y, transform=None, transform_2_pil=False):
        self.X = x
        self.Y = y
        self.transform = transform
        self.transform_2_pil = transform_2_pil

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]
        if self.transform_2_pil:
            image = self.to_pil_image(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    @staticmethod
    def to_pil_image(img_tensor):
        """Converte un tensore PyTorch o un array NumPy in un'immagine PIL RGB"""

        # Se è un tensore PyTorch, lo convertiamo in NumPy
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu().numpy()

        # Assicuriamoci che sia un array NumPy
        img_array = np.asarray(img_tensor)

        # Troviamo gli assi validi per height, width e channel
        possible_axes = [i for i in range(len(img_array.shape)) if img_array.shape[i] > 4]  # H e W sono almeno 28
        channel_axes = [i for i in range(len(img_array.shape)) if img_array.shape[i] in {1, 2, 3, 4}]

        if len(possible_axes) != 2:
            raise ValueError(f"Shape non valido per immagine: {img_array.shape}")

        # Assumiamo che il resto sia il canale
        channel_axis = [i for i in channel_axes if i not in possible_axes]
        if len(channel_axis) != 1:
            raise ValueError(f"Impossibile determinare il canale in {img_array.shape}")

        channel_axis = channel_axis[0]

        # Riordiniamo gli assi in (H, W, C)
        img_array = np.transpose(img_array, (possible_axes[0], possible_axes[1], channel_axis))

        # Convertiamo in uint8 se necessario
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)

        # Convertiamo in immagine PIL
        if img_array.shape[2] == 1:
            img_array = img_array.squeeze(-1)
            img = PIL.Image.fromarray(img_array, mode="L")
        else:
            img = PIL.Image.fromarray(img_array)

        # Se l'immagine non è già RGB, la convertiamo
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img


def prepare_data(ds_root: str, ds_name: str, num_clients: int = 1, seed: int = 42,
                 split_iid: bool = True, concentration: float = 0.5, medmnist_size: int = 224,
                 generate_distribution: bool = False, val_percentage: int = 10, strategy_name: str = "fedavg"):
    """
    This method is used to download a dataset and prepare its partitions
    :param ds_root:
    :param ds_name:
    :param num_clients:
    :param seed:
    :param split_iid:
    :param concentration:
    :param medmnist_size:
    :param generate_distribution:
    :param val_percentage:
    :param strategy_name:
    :return:
    """
    dataset = _download_dataset(ds_root, ds_name, medmnist_size, seed)
    if split_iid:
        train_distributions = _save_data_iid(ds_root, ds_name, dataset.train_set, dataset.test_set,
                                             val_percentage, num_clients, seed, medmnist_size,
                                             generate_distribution, dataset.get_num_classes(),
                                             dataset.get_task_name(), strategy_name)
    else:
        train_distributions = _save_data_non_iid(ds_root, ds_name, dataset.train_set, dataset.test_set,
                                                 val_percentage, num_clients, seed, medmnist_size,
                                                 concentration, generate_distribution, dataset.get_num_classes(),
                                                 dataset.get_task_name(), strategy_name)
    #
    return train_distributions, dataset.get_num_classes(), dataset.get_task_name()


def _download_dataset(ds_root: str, ds_name: str, medmnist_size: int = 224, seed: int = 42):
    match ds_name:
        case "cifar10" | "cifar100":
            dataset = get_cifar(ds_root, ds_name, seed)
        case "mnist" | "fashionmnist":
            dataset = get_mnist(ds_root, ds_name, seed)
        case "caltech256":
            dataset = get_caltech(ds_root, ds_name, seed)
        case "eurosat":
            dataset = get_eurosat(ds_root, ds_name, seed)
        case "imagenette":
            dataset = get_imagenette(ds_root, ds_name, seed)
        case _:
            if ds_name.startswith("medmnist_"):
                dataset = get_medmnist(ds_root, ds_name, seed, medmnist_size)
            else:
                assert False, f"Dataset '{ds_name}' not managed yet!"
    return dataset


def _save_data_iid(dataset_path: str, dataset_name: str, train_set: Dataset, test_set: Dataset,
                   val_percentage: int = 10, num_clients: int = 1, seed: int = 42, medmnist_size: int = 224,
                   generate_distribution: bool = False, num_classes: int = 10, task_name: str = "multi-class",
                   strategy_name: str = "fedavg"):
    dataset_fed_path = get_dataset_fed_path(dataset_path, dataset_name, medmnist_size=medmnist_size, seed=seed,
                                            val_size=val_percentage / 100, num_clients=num_clients, ds_iid=True,
                                            strategy_name=strategy_name)

    train_distributions = {}
    if os.path.exists(dataset_fed_path):
        print("IID partitions already available...")
        if generate_distribution:
            train_distributions = _get_partitions_train_distribution(dataset_fed_path, num_clients,
                                                                     num_classes, task_name)
        return train_distributions

    # If data is not in disk, we create the partitions and save them into disk
    print("Preparing IID partitions...")
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(train_set) // num_clients
    remainder = len(train_set) % num_clients
    lengths = [partition_size] * num_clients
    lengths[0] += remainder
    datasets = random_split(train_set, lengths, torch.Generator().manual_seed(seed))
    train_distributions = _split_datasets_and_save_partition(dataset_fed_path, datasets, test_set, val_percentage,
                                                             seed, num_classes, task_name, generate_distribution)
    print("Finished preparing IID partitions")
    return train_distributions


def _save_data_non_iid(dataset_path: str, dataset_name: str, train_set: Dataset, test_set: Dataset,
                       val_percentage: int = 10, num_clients: int = 1, seed: int = 42, medmnist_size: int = 224,
                       concentration: float = 0.1,
                       generate_distribution: bool = False, num_classes: int = 10, task_name: str = "multi-class",
                       strategy_name: str = "fedavg"):
    dataset_fed_path = get_dataset_fed_path(dataset_path, dataset_name, medmnist_size=medmnist_size,
                                            val_size=val_percentage / 100, seed=seed, num_clients=num_clients,
                                            concentration=concentration, ds_iid=False, strategy_name=strategy_name)
    train_distributions = {}
    if os.path.exists(dataset_fed_path):
        print("NON-IID partitions already available...")
        if generate_distribution:
            train_distributions = _get_partitions_train_distribution(dataset_fed_path, num_clients,
                                                                     num_classes, task_name)
        return train_distributions

    # If data is not in disk, we create the partitions and save them into disk
    print("Preparing NON-IID partitions...")
    # TRAIN DATA MANAGEMENT
    dataset_numpy = _dataset_to_numpy(train_set)
    test_set = _dataset_to_numpy(test_set)
    test_set = NumpyDataset(test_set[0], test_set[1], transform=transforms.ToTensor())
    partitions, distribution = create_lda_partitions(dataset_numpy, dirichlet_dist=None, num_partitions=num_clients,
                                                     concentration=concentration, seed=seed, accept_imbalanced=True)
    datasets = [NumpyDataset(partition[0], partition[1], transform=transforms.ToTensor()) for partition in partitions]
    train_distributions = _split_datasets_and_save_partition(dataset_fed_path, datasets, test_set, val_percentage,
                                                             seed, num_classes, task_name, generate_distribution)
    print("Finished preparing NON-IID partitions")
    return train_distributions


def _get_partitions_train_distribution(dataset_path: str, num_clients: int, num_classes: int, task_name: str):
    train_distributions = {}
    for current_client in range(num_clients):
        ds_train = load_partition(dataset_path, current_client, ["train"])["train"]
        train_distributions[current_client] = get_label_distribution(DataLoader(ds_train), num_classes, task_name)
    return train_distributions


def _split_datasets_and_save_partition(dataset_fed_path: str, datasets: list[Dataset], test_set: Dataset,
                                       val_percentage: int = 10, seed: int = 42, num_classes: int = 10,
                                       task_name: str = "multi-class", generate_distribution=False):
    train_distributions = {}
    for current_client, dataset in enumerate(datasets):
        len_val = len(dataset) // val_percentage  # 10 % validation set
        len_train = len(dataset) - len_val
        ds_lengths = [len_train, len_val]
        ds_train, ds_val = random_split(dataset, ds_lengths, torch.Generator().manual_seed(seed))
        _save_partition(dataset_fed_path, current_client, train_data=ds_train, val_data=ds_val, test_data=test_set)
        if generate_distribution:
            train_distributions[current_client] = get_label_distribution(DataLoader(ds_train), num_classes, task_name)
    return train_distributions


def extract_imgs_and_labels(dataset):
    """
    Itera su un dataset PyTorch e restituisce tutte le immagini e le etichette.
    Funziona con qualsiasi combinazione di ConcatDataset e Subset.
    """
    all_imgs, all_labels = [], []
    there_are_monochrome = False
    there_are_colors = False
    for image, _ in dataset:
        if len(image.shape) == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))

        # Se il numero di canali è 1, lo espandiamo a 3 canali
        if image.shape[-1] == 1:
            there_are_monochrome = True
        if image.shape[-1] != 1:
            there_are_colors = True
    is_dataset_mixed = there_are_monochrome and there_are_colors

    for image, label in dataset:

        if len(image.shape) == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))

        if is_dataset_mixed and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)  # Ripeti il canale su 3 dimensioni

        all_imgs.append(image.cpu().numpy())
        all_labels.append(np.array(label))

    # Converti le liste in array NumPy
    return np.array(all_imgs), np.array(all_labels)


def _save_partition(save_path: str, current_client: int,
                    train_data: Dataset, val_data: Dataset, test_data: Dataset):
    os.makedirs(save_path, exist_ok=True)
    # Train
    ds_train_imgs, ds_train_labels = extract_imgs_and_labels(train_data)
    np.savez(get_client_partition_path(save_path, current_client, "train"),
             data=ds_train_imgs, labels=ds_train_labels)
    # Validation
    ds_val_imgs, ds_val_labels = extract_imgs_and_labels(val_data)
    np.savez(get_client_partition_path(save_path, current_client, "validation"),
             data=ds_val_imgs, labels=ds_val_labels)
    # Test
    test_partition = get_client_partition_path(save_path, None, "test")
    if not os.path.exists(test_partition):
        ds_test_imgs, ds_test_labels = extract_imgs_and_labels(test_data)
        np.savez(test_partition, data=ds_test_imgs, labels=ds_test_labels)


def load_partition(load_path: str, current_client: int, partitions: list[str], transform_2_pil: bool = False):
    """
    This method is used to load a data partition from disk
    :param load_path:
    :param current_client:
    :param partitions:
    :param transform_2_pil:
    :return:
    """

    datasets = {}
    for partition in partitions:
        data = _load_partition_data(load_path, current_client, partition)
        datasets[partition] = NumpyDataset(data["data"], data["labels"],
                                           transform=transforms.ToTensor(), transform_2_pil=transform_2_pil)
    return datasets


def _load_partition_data(load_path: str, current_client: int, partition_name: str):
    if partition_name == "test":
        current_client = None
    partition_path = get_client_partition_path(load_path, current_client, partition_name)
    # Check the partition found in given path
    if not os.path.exists(partition_path):
        raise Exception("load_partition -> partition not found!")
    # Read data partition from disk and transform into a dataloader
    data = np.load(partition_path)
    return data


def delete_dataset(dataset_path: str, dataset_name: str, ds_iid: bool, val_percentage: int = 10,
                   num_clients: int = 1, seed: int = 42, medmnist_size: int = 224, strategy_name: str = "fedavg"):
    dataset_fed_path = get_dataset_fed_path(dataset_path, dataset_name, medmnist_size=medmnist_size, seed=seed,
                                            val_size=val_percentage / 100, num_clients=num_clients, ds_iid=ds_iid,
                                            strategy_name=strategy_name)
    _delete_partitions(dataset_fed_path)


def _delete_partitions(load_path: str):
    if not os.path.exists(load_path):
        print("_delete_partition -> partition not found! deleting nothing")
    else:
        shutil.rmtree(load_path)
        print("_delete_partition -> partition deleted!")


def load_loader(dataset: Dataset, batch_size: int, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _dataset_to_numpy(dataset):
    # Lista per memorizzare immagini e etichette
    all_images = []
    all_labels = []

    # Carica i dati in batch
    for image, label in dataset:
        # IF WE HAVE CHANNELS in the first position, then we move it into the last
        if len(image.shape) == 3 and image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))

        # Se il numero di canali è 1, lo espandiamo a 3 canali
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)  # Ripeti il canale su 3 dimensioni

        if isinstance(label, int):
            label = np.array([label])
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        all_images.append(image)
        all_labels.append(label)

    # Concatenare tutti i batch
    x = np.array(all_images)
    y = np.array(all_labels)
    return x, y


def load_aux_dataset(shape: tuple[int, int, int] = (3, 32, 32), aux_seed: int = 42, number_samples: int = 10,
                     batch_size: int = 32, transform=None, labels_size: int = 1):
    aux_set = RandomDataset(img_shape=shape, dataset_size=number_samples, seed=aux_seed, transform=transform,
                            labels_size=labels_size)
    aux_loader = DataLoader(aux_set, batch_size=batch_size, shuffle=False)
    return aux_loader


def get_dataset_fed_path(dataset_path: str, dataset_name: str,
                         medmnist_size: int = 28, val_size: float = 0.1, seed: int = 42, num_clients: int = 10,
                         concentration: float = 0.1, ds_iid: bool = False, strategy_name: str = "fedavg"):
    if ds_iid:
        return os.path.join(dataset_path, "iid", dataset_name, str(medmnist_size), str(val_size), str(num_clients),
                            f"seed_{str(seed)}", strategy_name)
    else:
        return os.path.join(dataset_path, "non_iid", dataset_name, str(medmnist_size), str(val_size), str(num_clients),
                            f"seed_{str(seed)}", f"beta_{str(concentration)}", strategy_name)


def get_client_partition_path(dataset_fed_path, current_client, partition_name):
    if current_client is None:
        return os.path.join(dataset_fed_path, f"partition_{partition_name}.npz")
    return os.path.join(dataset_fed_path, f"partition_{current_client}_{partition_name}.npz")


def load_transform(dataset_name: str, model_name: str, aux_transform: bool = False):
    final_img_size = 224
    if aux_transform:
        return transforms.Compose([transforms.Resize(final_img_size), transforms.ToTensor()])
    else:
        if "deit_vit" in model_name:
            final_img_size = 224
        match dataset_name:
            case "cifar10" | "cifar100":
                return get_cifar_transforms(final_size=final_img_size)
            case "mnist" | "fashionmnist":
                return get_mnist_transforms(dataset_name, final_size=final_img_size)
            case "caltech256":
                return get_caltech_transforms(final_size=final_img_size)
            case "eurosat":
                return get_eurosat_transforms(final_size=final_img_size)
            case "imagenette":
                return get_imagenette_transforms(final_size=final_img_size)
            case _:
                if dataset_name.startswith("medmnist_"):
                    return get_medmnist_transforms(final_size=final_img_size)
                else:
                    assert False, f"Transform for dataset '{dataset_name}' not managed yet!"

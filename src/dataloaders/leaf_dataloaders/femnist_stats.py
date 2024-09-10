from torchvision import transforms

from dataloaders.leaf_dataloaders import FEmnistDataset


def get_stats(dataset):
    # Definisci una trasformazione per convertire le immagini in tensor
    transform = transforms.ToTensor()

    # Inizializza i valori per la somma e il quadrato della somma
    mean = 0.0
    std = 0.0
    total_images = 0
    n_images = len(dataset)
    # Itera attraverso tutte le immagini nel folder
    for image, label in dataset:
        # Applica la trasformazione
        image = transform(image)

        # Calcola il numero di immagini
        total_images += 1

        # Aggiorna mean e std
        mean += image.mean([1, 2])
        std += image.std([1, 2])

        if total_images % 10000 == 0:
            print(total_images, "/", n_images)

    # Calcola la media e la std totali
    mean /= total_images
    std /= total_images
    return mean, std


if __name__ == "__main__":
    ds = FEmnistDataset("../../../datasets/leaf/")
    # mean, std:  (tensor([0.9627, 0.9627, 0.9627]), tensor([0.1550, 0.1550, 0.1550]))
    print("mean, std: ", get_stats(ds))

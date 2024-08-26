import os
from PIL import Image
from torchvision import transforms

# Definisci il percorso delle immagini
image_folder = 'C:/Users/Gianluca/Downloads/celeba/images'

# Definisci una trasformazione per convertire le immagini in tensor
transform = transforms.ToTensor()

# Inizializza i valori per la somma e il quadrato della somma
mean = 0.0
std = 0.0
total_images = 0

# Itera attraverso tutte le immagini nel folder
for image_name in os.listdir(image_folder):
    # Carica l'immagine
    img_path = os.path.join(image_folder, image_name)
    image = Image.open(img_path)

    # Applica la trasformazione
    image = transform(image)

    # Calcola il numero di immagini
    total_images += 1

    # Aggiorna mean e std
    mean += image.mean([1, 2])
    std += image.std([1, 2])

    if total_images % 1000 == 0:
        print(total_images)

# Calcola la media e la std totali
mean /= total_images
std /= total_images

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")

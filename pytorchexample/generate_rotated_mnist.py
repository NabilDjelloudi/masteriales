import os
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, functional as TF
from torch.utils.data import Subset
import random

def download_mnist(data_dir="./pytorchexample/data"):
    """Télécharge le dataset MNIST."""
    train_dataset = MNIST(root=data_dir, train=True, download=True)
    test_dataset = MNIST(root=data_dir, train=False, download=True)
    return train_dataset, test_dataset

def apply_rotation(dataset, angle):
    """Applique une rotation à chaque image du dataset."""
    rotated_data = []
    for img, label in dataset:
        rotated_img = TF.rotate(img, angle)
        rotated_data.append((ToTensor()(rotated_img), label))  # Convertir en tenseur directement
    return rotated_data

def create_partitions(dataset, num_clients, angles, seed=42):
    """Crée des partitions tournées pour chaque client."""
    if len(angles) != num_clients:
        raise ValueError("Le nombre d'angles doit correspondre au nombre de clients.")

    random.seed(seed)
    partition_size = len(dataset) // num_clients
    partitions = []
    for client_id in range(num_clients):
        start_idx = client_id * partition_size
        end_idx = start_idx + partition_size
        subset = Subset(dataset, range(start_idx, end_idx))
        rotated_subset = apply_rotation(subset, angles[client_id])
        partitions.append(rotated_subset)
    return partitions

def save_partitions(partitions, output_dir="./pytorchexample/data/rotated_mnist"):
    """Sauvegarde les partitions de données dans des fichiers .pt."""
    os.makedirs(output_dir, exist_ok=True)
    for client_id, partition in enumerate(partitions):
        # Diviser en données d'entraînement (80%), de validation (10%) et de test (10%)
        train_split = int(0.8 * len(partition))
        val_split = int(0.9 * len(partition))  # 80% train, 10% val, 10% test

        train_partition = partition[:train_split]
        val_partition = partition[train_split:val_split]
        test_partition = partition[val_split:]

        # Sauvegarder les données
        torch.save(train_partition, os.path.join(output_dir, f"client_{client_id}_train.pt"))
        torch.save(val_partition, os.path.join(output_dir, f"client_{client_id}_val.pt"))
        torch.save(test_partition, os.path.join(output_dir, f"client_{client_id}_test.pt"))
    print(f"Partitions sauvegardées dans {output_dir}.")

def generate_rotated_mnist(num_clients=10, angles=None, data_dir="./pytorchexample/data"):
    """Pipeline complet pour générer les partitions tournées MNIST."""
    print("Téléchargement du dataset MNIST...")
    train_dataset, _ = download_mnist(data_dir)

    print("Création des partitions tournées...")
    if angles is None:
        # Générer des angles si non spécifiés
        angles = [(i * 360) // num_clients for i in range(num_clients)]
    partitions = create_partitions(train_dataset, num_clients, angles)

    print("Sauvegarde des partitions...")
    save_partitions(partitions)

    print("Partitions générées et sauvegardées avec succès.")

if __name__ == "__main__":
    # Configurer les paramètres ici
    NUM_CLIENTS = 10
    ANGLES = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]  # Angles de rotation pour chaque client
    DATA_DIR = "./pytorchexample/data"

    generate_rotated_mnist(num_clients=NUM_CLIENTS, angles=ANGLES, data_dir=DATA_DIR)

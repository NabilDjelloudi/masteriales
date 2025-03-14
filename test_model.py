import torch
import pickle
import argparse
import os
from pytorchexample.task import Net, test, load_data

# Parser pour les arguments
parser = argparse.ArgumentParser(description="Test d'un modèle fédéré")
parser.add_argument("--strategy", type=str, required=True, help="Nom de la stratégie (fedavg, fedprox, fednova)")
parser.add_argument("--round", type=int, required=True, help="Numéro du round à tester")
parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu' ou 'cuda'")

args = parser.parse_args()

# Définir l'appareil (CPU ou GPU)
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

#  Charger le modèle
model_path = f"saved_models/{args.strategy}/global_parameters_round_{args.round}.pth"
print(f" Chargement du modèle {args.strategy} après {args.round} rounds sur {device}...")

try:
    model = Net().to(device)  # Instancier le modèle et l'envoyer sur le bon device
    state_dict = torch.load(model_path, map_location=device)  # Charger les poids sauvegardés
    model.load_state_dict(state_dict)  # Appliquer les poids au modèle
    model.eval()  # Mettre le modèle en mode évaluation
    print("✅ Modèle chargé avec succès !")

    #  Charger les données de test (Modifie `partition_id` si nécessaire)
    _, testloader = load_data(partition_id=0, num_partitions=10, batch_size=32, rotation_degree=0)

    #  Tester le modèle
    test_loss, accuracy = test(model, testloader, device)
    print(f" Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2%}")

except FileNotFoundError:
    print(f" Fichier non trouvé : {model_path}")
except RuntimeError as e:
    print(f" Erreur lors du chargement du modèle : {e}")
except Exception as e:
    print(f" Erreur inattendue : {e}")

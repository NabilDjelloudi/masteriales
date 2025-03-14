"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
import time  # Import pour mesurer le temps
from flwr.common import Context

from pytorchexample.task import Net, get_weights, load_data, set_weights, test, train


# Définir les degrés de rotation pour chaque client
ROTATION_DEGREES = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, proximal_mu=0.0):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)  # Déplacement du modèle sur le dispositif
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.proximal_mu = proximal_mu  # Facteur proximal pour FedProx


    def fit(self, parameters, config):
        """Train the model with data of this client."""
        print(f"Received weights: {len(parameters)} layers")
        set_weights(self.net, parameters)  # Appliquer les poids reçus

        start_time = time.time()
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            proximal_mu=self.proximal_mu if config.get("strategy", "") == "FedProx" else 0.0,  #  Appliquer FedProx si activé
            global_weights=parameters if config.get("strategy", "") == "FedProx" else None,  #  Poids globaux pour régularisation
        )

        end_time = time.time()
        training_time = end_time - start_time

        results["time"] = training_time  #  Ajouter le temps d'entraînement

        #  Tester après l'entraînement
        loss, accuracy = test(self.net, self.valloader, self.device)
        results["accuracy"] = accuracy  #  Ajouter accuracy

        # 🚀 Vérifier ce que le client envoie au serveur
        print(f"[Client] Sending results: {results}")

        return get_weights(self.net), len(self.trainloader.dataset), results




    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construit un client qui choisit la bonne fonction d'entraînement selon la stratégie."""
    
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    assert partition_id < len(ROTATION_DEGREES), "Partition ID out of range for ROTATION_DEGREES"

    rotation_degree = ROTATION_DEGREES[partition_id]
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size, rotation_degree)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    proximal_mu = context.run_config.get("proximal-mu", 0.0)  #  Récupérer mu si FedProx
    client = FlowerClient(trainloader, valloader, local_epochs, learning_rate, proximal_mu)


    return client.to_client()





def evaluate(self, parameters, config):
    """Evaluate the model on the data this client has."""
    if not parameters:
        raise ValueError("Received empty parameters during evaluation.")
    print(f"Received weights for evaluation: {len(parameters)} layers")
    set_weights(self.net, parameters)
    loss, accuracy = test(self.net, self.valloader, self.device)
    return loss, len(self.valloader.dataset), {"accuracy": accuracy}


# Flower ClientApp
app = ClientApp(client_fn)

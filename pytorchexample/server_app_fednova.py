"""Server app for Flower PyTorch example with FedNova."""

import pickle
from pathlib import Path
import os
import torch
from typing import List, Tuple


from flwr.common import Parameters, Metrics, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

from pytorchexample.task import Net, get_weights

# Répertoire pour sauvegarder les poids
output_dir = Path("./saved_models/fednova")
output_dir.mkdir(parents=True, exist_ok=True)


class CustomFedNova(FedAvg):
    """FedNova strategy with temporal normalization and additional metrics."""
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        self.output_dir = "saved_models/fednova"
        os.makedirs(self.output_dir, exist_ok=True)  #  Assurer que le dossier existe

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate and save the model weights correctly."""
        if not results:
            print("[FedNova] No results received from clients.")
            return None, {}

        #  Normalisation temporelle
        total_time = sum(fit_res.metrics.get("time", 1.0) for _, fit_res in results)
        normalized_weights = []

        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            time_ratio = fit_res.metrics.get("time", 1.0) / max(1e-6, total_time)
            normalized_weights.append([w * time_ratio for w in weights])

        # Agréger les poids normalisés
        aggregated_weights = [
            sum(weight[k] for weight in normalized_weights) for k in range(len(normalized_weights[0]))
        ]

        # Sauvegarde correcte avec `torch.save`
        if server_round == self.num_rounds:
            try:
                model = Net()  # Instancier le modèle
                state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), aggregated_weights)}
                output_path = f"{self.output_dir}/global_parameters_round_{server_round}.pth"
                torch.save(state_dict, output_path)  # Sauvegarde PyTorch correcte
                print(f" Poids globaux FedNova sauvegardés dans {output_path}")
            except Exception as e:
                print(f" [FedNova] Erreur lors de la sauvegarde des poids : {e}")

        #  Retourner les poids et les métriques globales
        avg_loss = sum(fit_res.metrics["loss"] for _, fit_res in results) / len(results)
        global_accuracy = sum(fit_res.metrics.get("accuracy", 0.0) for _, fit_res in results) / len(results)

        global_metrics = {
            "global_loss": avg_loss,
            "global_accuracy": global_accuracy,
        }

        return ndarrays_to_parameters(aggregated_weights), global_metrics

def server_fn(context: dict) -> ServerAppComponents:
    """Initialiser et retourner les composants du serveur."""
    num_rounds = context.run_config.get("num-server-rounds", 3)

    net = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(net))

    strategy = CustomFedNova(
        num_rounds=num_rounds,
        fraction_fit=1.0,
        min_fit_clients=10,
        min_available_clients=10,
        initial_parameters=initial_parameters,
    )

    print(f"[FedNova] Strategy configured for {num_rounds} rounds")

    config = ServerConfig(num_rounds=num_rounds)

    # Ajouter l'indication de stratégie pour les clients
    context.run_config["strategy"] = "FedNova"

    return ServerAppComponents(strategy=strategy, config=config)



# Créer l'application du serveur
app = ServerApp(server_fn=server_fn)

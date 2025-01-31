"""Server app for Flower PyTorch example with FedNova."""

import pickle
from pathlib import Path
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

    def __init__(
        self,
        total_rounds: int,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 10,
        min_available_clients: int = 10,
        initial_parameters: Parameters = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )
        self.total_rounds = total_rounds

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Metrics]:
        """Aggregate fit results using FedNova logic with temporal normalization and display accuracy."""
        if not results:
            print("[FedNova] No results received from clients during aggregation.")
            return None, {}

        # ✅ Calculer les pondérations basées sur les durées locales (normalisation temporelle)
        total_time = sum(fit_res.metrics.get("time", 1.0) for _, fit_res in results)
        normalized_weights = []

        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            time_ratio = fit_res.metrics.get("time", 1.0) / max(1e-6, total_time)  # Évite KeyError et division par zéro
            normalized_weights.append([w * time_ratio for w in weights])

        # ✅ Agréger les poids normalisés
        aggregated_weights = [
            sum(weight[k] for weight in normalized_weights) for k in range(len(normalized_weights[0]))
        ]

        # ✅ Calculer la **moyenne des accuracies et des pertes**
        avg_loss = sum(fit_res.metrics["loss"] for _, fit_res in results) / len(results)
        global_accuracy = sum(fit_res.metrics.get("accuracy", 0.0) for _, fit_res in results) / len(results)

        # ✅ Afficher `Accuracy` **correctement** après chaque round
        print(f"[FedNova] Round {server_round}: Loss={avg_loss:.4f}, Accuracy={global_accuracy:.4f}")

        # ✅ Sauvegarde des poids si c'est le dernier round
        if server_round == self.total_rounds:
            try:
                with open(f"{output_dir}/global_parameters_round_{server_round}.pkl", "wb") as f:
                    pickle.dump(aggregated_weights, f)
                print(f"Poids globaux finaux sauvegardés dans {output_dir}/global_parameters_round_{server_round}.pkl")
            except Exception as e:
                print(f"[FedNova] Error saving final weights: {e}")

        # ✅ Retourner les poids et **les métriques globales**
        global_metrics = {
            "global_loss": avg_loss,
            "global_accuracy": global_accuracy,  # ✅ Ajout d'Accuracy dans les métriques
        }

        return ndarrays_to_parameters(aggregated_weights), global_metrics


def server_fn(context: dict) -> ServerAppComponents:
    """Initialiser et retourner les composants du serveur."""
    num_rounds = context.run_config.get("num-server-rounds", 3)

    net = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(net))

    strategy = CustomFedNova(
        total_rounds=num_rounds,
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

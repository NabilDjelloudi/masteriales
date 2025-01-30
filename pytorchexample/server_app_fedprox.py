"""Server app for FedProx with Flower."""

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
output_dir = Path("./saved_models/fedprox")
output_dir.mkdir(parents=True, exist_ok=True)


class FedProx(FedAvg):
    """FedProx strategy with proximal regularization."""

    def __init__(
        self,
        proximal_mu: float = 1.0,  # Facteur proximal
        total_rounds: int = 3,  # Nombre total de rounds
        initial_parameters: Parameters = None,  # Paramètres initiaux
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu
        self.total_rounds = total_rounds
        self.initial_parameters = initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, Metrics]:
        """Aggregate weights with FedProx logic."""
        if not results:
            print("No results received from clients during aggregation.")
            return None, {}

        # Agrégation classique des poids
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        weighted_updates = []

        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            weighted_updates.append([w * fit_res.num_examples for w in weights])

        aggregated_weights = [
            sum(weight[k] for weight in weighted_updates) / total_examples
            for k in range(len(weighted_updates[0]))
        ]

        # Ajouter le terme proximal (FedProx)
        if self.proximal_mu > 0 and self.initial_parameters is not None:
            global_weights = parameters_to_ndarrays(self.initial_parameters)
            aggregated_weights = [
                w + self.proximal_mu * (global_w - w)
                for w, global_w in zip(aggregated_weights, global_weights)
            ]
            print(f"[FedProx] Applied proximal term with mu={self.proximal_mu} at round {server_round}")

        # Sauvegarde des poids si c'est le dernier round
        if server_round == self.total_rounds:
            with open(f"{output_dir}/global_parameters_round_{server_round}.pkl", "wb") as f:
                pickle.dump(aggregated_weights, f)
            print(f"Poids globaux finaux sauvegardés dans {output_dir}/global_parameters_round_{server_round}.pkl")

        return ndarrays_to_parameters(aggregated_weights), {}


def server_fn(context: dict) -> ServerAppComponents:
    """Configurer FedProx pour le serveur Flower."""
    num_rounds = context.run_config.get("num-server-rounds", 3)
    mu = context.run_config.get("proximal-mu", 1.0)

    # Charger le modèle pour obtenir les paramètres initiaux
    model = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(model))

    # Configuration de la stratégie
    strategy = FedProx(
        proximal_mu=mu,
        total_rounds=num_rounds,
        fraction_fit=context.run_config.get("fraction-fit", 1.0),
        min_fit_clients=context.run_config.get("min-fit-clients", 2),
        min_available_clients=context.run_config.get("min-available-clients", 2),
        initial_parameters=initial_parameters,
    )

    print(f"[FedProx] Strategy configured with mu={mu} for {num_rounds} rounds")
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Déclarez l'application
app = ServerApp(server_fn=server_fn)

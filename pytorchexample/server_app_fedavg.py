"""pytorchexample: A Flower / PyTorch app."""

from pathlib import Path
import numpy as np

from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import torch

from pytorchexample.task import Net, get_weights

# Répertoire pour sauvegarder les poids
output_dir = Path("./saved_models/fedavg")
output_dir.mkdir(parents=True, exist_ok=True)

# Fonction d'agrégation des métriques
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_metrics_aggregation(metrics):
    try:
        losses = [m[1]["loss"] for m in metrics]
        return {"loss": sum(losses) / len(losses)}
    except Exception as e:
        print(f"Erreur dans fit_metrics_aggregation : {e}")
        return {}

# Stratégie personnalisée
class SaveFinalWeightsFedAvg(FedAvg):
    def __init__(self, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate and directly save the parameters."""
        try:
            aggregated_result = super().aggregate_fit(server_round, results, failures)

            #  Vérification du type de sortie
            if isinstance(aggregated_result, tuple):  
                aggregated_parameters, metrics_aggregated = aggregated_result  # On récupère les poids et les métriques
            else:
                aggregated_parameters = aggregated_result  
                metrics_aggregated = {}  # On met un dict vide pour éviter les erreurs

            # Convertir en liste de tenseurs PyTorch
            aggregated_weights = list(parameters_to_ndarrays(aggregated_parameters))

            # Convertir en state_dict pour PyTorch
            model = Net()  # Instancier le modèle
            state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), aggregated_weights)}

            # Sauvegarde correcte
            output_path = output_dir / f"global_parameters_round_{server_round}.pth"
            torch.save(state_dict, output_path)
            print(f" Poids globaux sauvegardés dans {output_path}")

            return aggregated_parameters, metrics_aggregated  #  On retourne bien un tuple

        except Exception as e:
            print(f" Erreur lors de la sauvegarde des paramètres : {e}")
            return None, None  # Éviter que le serveur plante


def server_fn(context: Context) -> ServerAppComponents:
    """Configurer les composants du serveur Flower."""
    num_rounds = context.run_config["num-server-rounds"]

    # Initialiser les paramètres du modèle
    model = Net()
    initial_weights = get_weights(model)
    initial_parameters = ndarrays_to_parameters(initial_weights)

    # Définir la stratégie avec sauvegarde
    strategy = SaveFinalWeightsFedAvg(
        num_rounds=num_rounds,
        fraction_fit=1.0,
        fraction_evaluate=context.run_config["fraction-evaluate"],
        min_available_clients=2,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Créer l'application du serveur
app = ServerApp(server_fn=server_fn)

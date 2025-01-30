import sys
import toml
from pathlib import Path
import os

PYPROJECT_PATH = Path("pyproject.toml")

def update_pyproject(algorithm: str):
    """Met à jour pyproject.toml pour l'algorithme spécifié."""
    if not PYPROJECT_PATH.exists():
        print("[ERREUR] pyproject.toml introuvable.")
        sys.exit(1)

    pyproject = toml.load(PYPROJECT_PATH)

    # Modifier le serverapp en fonction de l'algorithme
    if algorithm == "fedavg":
        pyproject["tool"]["flwr"]["app"]["serverapp"] = pyproject["tool"]["flwr"]["app"]["components"]["fedavg_serverapp"]
    elif algorithm == "fednova":
        pyproject["tool"]["flwr"]["app"]["serverapp"] = pyproject["tool"]["flwr"]["app"]["components"]["fednova_serverapp"]
    elif algorithm == "fedprox":
        pyproject["tool"]["flwr"]["app"]["serverapp"] = pyproject["tool"]["flwr"]["app"]["components"]["fedprox_serverapp"]
    else:
        print(f"[ERREUR] Algorithme inconnu : {algorithm}")
        sys.exit(1)

    # Sauvegarder les changements dans pyproject.toml
    with open(PYPROJECT_PATH, "w") as f:
        toml.dump(pyproject, f)

    print(f"[SUCCÈS] pyproject.toml mis à jour pour utiliser {algorithm}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python run_flower.py <algorithm>")
        sys.exit(1)

    algorithm = sys.argv[1].lower()
    update_pyproject(algorithm)

    # Lancer Flower avec le nouvel algorithme
    exit_code = os.system("flwr run .")
    sys.exit(exit_code)

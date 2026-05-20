# ONERA 468 CRM Challenge rho : Starting Kit

## Objectif

Prédire la densité volumique adimensionnée $\rho$ en chaque point du domaine fluide,
pour des conditions aérodynamiques non vues à l'entraînement.

## Structure des données

| Fichier | Shape | Description |
|---|---|---|
| `X9_TRAIN_POINT_fl32.npy` | (81 361 488, 9) | Inputs train |
| `RHO_TRAIN_POINT_fl32.npy` | (81 361 488, 1) | Cible train (ρ) |
| `X9_TEST_POINT_fl32.npy` | (40 680 744, 9) | Inputs test |

Les 9 colonnes de X sont : `x, y, z, nx, ny, nz, Minf, AoA, pi`

- 260 774 points par condition aérodynamique
- 312 conditions en train, 156 en test
- `pi` est le facteur de pression de stagnation (1, 2 ou 4)

## Soumission

Votre soumission doit être un fichier zip contenant `model.py` avec une classe `Model`
implémentant `fit(X, y)` et `predict(X)`.

## Métriques

- **R^2** (pondéré par confidence score) plus proche de 1 est mieux
- **wrMAE** (worst-case relative MAE sur les cas à confidence=1) plus proche de 0 est mieux
- **score** = 5 × R^2 + 5 × (1 − wrMAE) métrique principale du leaderboard

## Lancer le starting kit

```bash
pip install -r requirements.txt
jupyter notebook starting_kit.ipynb
```
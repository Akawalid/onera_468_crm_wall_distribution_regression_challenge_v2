# ONERA 468 CRM Challenge rho : Starting Kit

## Objectif

Prédire la densité volumique adimensionnée $\rho$ en chaque point du domaine fluide,
pour des conditions aérodynamiques non vues à l'entraînement.

## Structure des données (`input_data/`, téléchargé depuis Codabench)

| Fichier | Shape | Description |
|---|---|---|
| `train_data.npy` | (n_train × 260 774, 9) | Inputs train |
| `train_labels.npy` | (n_train × 260 774, 1) | Cible train (ρ) |
| `test_data.npy` | (n_test × 260 774, 9) | Inputs test (sans labels) |
| `component_labels_unique.npy` | (260 774,) | Id de composant par point de paroi |
| `component_map.json` | -- | `{id: nom_composant}` (wing/pylon/fuselage/nacelle) |

Les 9 colonnes de X sont : `x, y, z, nx, ny, nz, Minf, AoA, Pi`

- 260 774 points de paroi par condition aérodynamique
- `Pi` est le facteur de pression de stagnation (1, 2 ou 4)
- **Vous n'avez jamais accès aux labels de test** (`reference_data/`) -- tout le
  notebook évalue les modèles par validation croisée sur `train` uniquement.

## Contenu du kit

- `starting_kit.ipynb` -- le notebook principal, à lire dans l'ordre.
- `kit_utils/` -- tout le code réutilisable (métriques, modèles, plots), importé
  par le notebook. Regardez-y directement si vous voulez comprendre ou modifier
  un détail d'implémentation.
  - `metrics.py` -- R2, wrMAE, KLw (mean_KL) et leurs intervalles de confiance
    bootstrap, formules identiques à `scoring_program/scoring.py`.
  - `data.py` -- chargement des données train et des masques de composants,
    découpage en folds de validation croisée (leave-two-Machs-out).
  - `lgbm_baseline.py` -- baseline simple : LightGBM pointwise.
  - `mlp_klw.py` -- baseline principale : MLP full-field entraîné avec une
    perte KL différentiable (remplace l'ancienne baseline KNN).
  - `pca_plots.py` -- diagnostics visuels (erreur sur l'avion, directions PCA
    de l'erreur, coupes).

## Soumission

Votre soumission doit être un fichier zip contenant `model.py` (à la racine du
zip, pas dans un sous-dossier) avec une classe `Model` implémentant `fit(X, y)`
et `predict(X)`. Le notebook génère `submission/model.py` (version autonome de
la baseline MLP) et `submission.zip` en section 5.

## Métriques (voir `bundle/scoring_program/scoring.py` pour le code officiel)

- **KLw** (`mean_KL`) -- **métrique principale du leaderboard**, plus proche de
  0 est mieux. Mesure la distance (KL-divergence) entre la distribution de vos
  résidus et une gaussienne de référence étroite, pondérée par composant
  (wing/pylon 0.3, fuselage/nacelle 0.2). Voir section 2.2 du notebook.
- **R^2** (pondéré par confidence score) -- plus proche de 1 est mieux.
- **wrMAE** (worst-case relative MAE sur les cas à confidence=1) -- plus proche
  de 0 est mieux.
- `score` = 5 × R² + 5 × (1 − wrMAE) -- conservé pour référence, KLw reste la
  métrique triée sur le leaderboard.

## Lancer le starting kit

```bash
pip install -r requirements.txt
jupyter notebook starting_kit.ipynb
```

La cross-validation complète (les deux baselines) tourne en environ 5 minutes
sur CPU.

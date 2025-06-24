# Modèles de Test — bobodiff

Ce dossier contient des modèles développés pour tester les capacités de la bibliothèque `bobodiff` en différentiation automatique, aussi bien sur des cas simples que sur des réseaux de neurones plus complexes.

---

## Contenu

### `predire_note.py` – Régression linéaire pour prédire des notes de concours

Ce script implémente une **régression linéaire** avec calcul du gradient par différentiation automatique grâce à `bobodiff`.

- **Objectif** : prédire les notes d’un concours à partir de différentes variables.
- **Méthode** : descente de gradient utilisant bobodiff (mode backward).
- **Auteur** : BOUEKE Omer Bokassa
- **Inspiré d’un projet pédagogique de** :
  
  > Mr. Laurent RISSER, PhD  
  > CNRS – Institut de Mathématiques de Toulouse (UMR 5219)  
  > ANITI – Artificial and Natural Intelligence Toulouse Institute

---

### `mlp_classe.py` – Mini réseau de neurones (MLP)

Un **perceptron multicouche** inspiré du travail d’Andrej Karpathy, utilisé pour tester `bobodiff` dans un contexte plus complexe que la régression linéaire.

- Réseau dense avec fonction d’activation et rétropropagation manuelle.
- Données jouets pour la classification.

---

### `mlp_book.py` – Réseau de neurones inspiré du projet `micrograd`

Un second test sur un **MLP minimaliste** également basé sur l’approche de Karpathy.

- Structure modulaire simplifiée pour tester `bobodiff`.
- Ce script imite les étapes d’apprentissage sur un mini dataset.

---

## Objectif du dossier

Ces fichiers servent à **valider l’efficacité et la flexibilité de `bobodiff`** sur différents types de modèles :

- Cas simple : régression linéaire
- Cas complexe : réseaux de neurones multicouches

Ils permettent aussi de comparer les résultats obtenus avec bobodiff à ceux issus de calculs manuels ou d’implémentations classiques.

---

## Remarques

- Tous les modèles utilisent des gradients calculés automatiquement via `bobodiff`.
- Ce travail a une vocation **expérimentale et pédagogique**.

```` Bash
python3 predire_note.py
python3 mlp_book.py
python3 mlp_classe.py
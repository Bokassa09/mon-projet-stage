# Deep Learning: Module de Différenciation Automatique en Python

## Description du Projet

Ce projet est une implémentation d'un module de différenciation automatique en Python utilisant la surcharge d'opérateur. Le module prend en charge deux modes :

Mode Forward (propagation directe) : calcule les dérivées en suivant les opérations de manière séquentielle.

Mode Backward (propagation inverse) : construit un graphe computationnel pour calculer les dérivées en partant de la sortie.

La différenciation automatique est une technique essentielle en optimisation et apprentissage automatique, offrant des dérivées exactes avec un coût calculatoire maîtrisé. Ce module constitue une base solide pour comprendre les concepts fondamentaux des réseaux de neurones et du deep learning.

## Structure du Projet

```
stage/
├── bobodiff/                    # 📚 Module principal de différenciation automatique
│   ├── backward/               # Implémentation du mode backward (propagation inverse)
│   ├── forward/                # Implémentation du mode forward (propagation directe)
│   └── utile.py               # Fonctions utilitaires communes
├── benchmark/                  # 📊 Évaluation et comparaison des performances
│   ├── profile_backward.py     # Profiling du mode backward
│   ├── profile_forward.py      # Profiling du mode forward
│   └── pytorch.py             # Comparaisons avec PyTorch
├── exemple/                    # 🧪 Exemples d'application pratique
│   ├── exemple_backward.py     # Tests des fonctions du rapport (mode backward)
│   └── exemple_forward.py      # Tests des fonctions du rapport (mode forward)
├── Modele/                     # 🧠 Applications sur réseaux de neurones
│   ├── predire_note.py         # Modèle de prédiction de notes
│   ├── mlp_book.py            # Perceptron multicouche (théorie)
│   └── mlp_classe.py          # MLP de classification binaire
├── Rapport_de_stage/           # 📄 Documentation et rapport
│   ├── [Fichiers PDF]         # Fondements théoriques et mathématiques
│   └── soutenance.pptx        # Présentation de soutenance
└── test/                       # ✅ Suite de tests de validation
|    ├── test_backward.py        # Tests complets du mode backward
|    └── test_forward.py 
|   # Tests complets du mode forward
|
└── Docs/                      # 🐍 Guide d'utilisation
    ├── usage.txt               # Guide pour les deux modes 
```

### 📖 Guide de navigation

**⚠️ Important :** Chaque répertoire contient son propre README avec des instructions détaillées. Il est **fortement recommandé** de consulter le README spécifique de chaque dossier avant d'utiliser les fichiers qu'il contient.

### Répertoires principaux

- **`bobodiff/`** : Cœur du module avec les implémentations des modes forward et backward
- **`benchmark/`** : Scripts de mesure de performance et comparaisons avec des bibliothèques établies
- **`exemple/`** : Démonstrations pratiques des fonctionnalités sur les cas d'étude du rapport
- **`Modele/`** : Applications du module sur des architectures de réseaux de neurones
- **`test/`** : Tests exhaustifs validant toutes les opérations mathématiques implémentées
- **`Rapport_de_stage/`** : Documentation théorique complète et présentation finale
- **`Docs/`** : Documentation claire d’utilisation de la bibliothèque bobodiff



## Objectifs

Étudier les principes de la différenciation automatique (DA).

Concevoir une architecture modulaire en Python avec surcharge d’opérateur.

Implémenter le mode Forward (propagation directe des dérivées).

Implémenter le mode Backward (propagation inverse avec graphe computationnel).

Tester sur des fonctions mathématiques et des modèles d'apprentissage.

Explorer l'utilisation de la différenciation automatique dans les réseaux de neurones.

Comparer les performances avec des bibliothèques comme JAX, TensorFlow et PyTorch.

## Applications en Deep Learning

Ce module peut être utilisé comme base pour comprendre et expérimenter avec les réseaux de neurones, notamment dans la mise en œuvre de la rétropropagation (backpropagation) utilisée dans l'apprentissage des modèles de deep learning.

## Génération automatique de docstrings

Ce projet utilise l'extension Python Docstring Generator pour **VS Code**, permettant de générer automatiquement des docstrings conformes aux standards (Google, NumPy, etc.).

Cela facilite la documentation des fonctions et classes, et garantit une meilleure lisibilité et maintenabilité du code.

## Prérequis

Python 3.8+

## Auteur

**BOUEKE Omer Bokassa** - Projet réalisé dans le cadre d'un stage supervisé par M. David DEFOUR.

**M. David DEFOUR**
Professeur d'informatique à l'Université de Perpignan et Vice-président chargé de la stratégie numérique et de l'intelligence artificielle.


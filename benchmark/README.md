# Benchmark & Profiling

Ce dossier contient les scripts et outils de benchmark pour évaluer les performances du mode **forward** (et bientôt **backward**) de la bibliothèque `bobodiff`.

---

## Contenu

- `profile_forward.py` : Script de profilage pour le mode forward (utilise `cProfile`).
- `forward.prof` : Résultat du profilage à analyser avec `snakeviz`.
- `profile.html` : Rapport Scalene interactif pour le mode forward
- `python3 -m http.server 8000` : Permet d'ouvrir le rapport HTML via un navigateur local.

---

## Profiling avec Scalene

1. **Installer Scalene** :

   ```bash
   pip install scalene

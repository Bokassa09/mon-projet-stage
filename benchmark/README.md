# Benchmark & Profiling

Ce dossier contient les scripts et outils de benchmark permettant d’évaluer les performances de la bibliothèque `bobodiff`, en particulier pour les modes **forward** et **backward** de la différentiation automatique.

---

## Contenu

- `profile_forward.py` : Script de profilage pour le mode forward (`cProfile`).
- `profile_backward.py` : Script de profilage pour le mode backward (`cProfile`).
- `forward.prof`, `backward.prof` : Résultats du profilage à analyser avec `snakeviz` 
- `profile.html`, `profile.html` : Rapports interactifs générés avec **Scalene**.

---

## Exécution du profilage

Avec `cProfile`

```bash
python profile_forward.py
python profile_backward.py

Avec `Snakeviz`

Visualisation avec snakeviz 

```bash
snakeviz forward.prof
snakeviz backward.prof

Avec `Scalene`

Visualisation de scalene

```bash

scalene profile_backward.py  # Génère un rapport HTML // Mode backward
python3 -m http.server 8000  # Pour visualiser le rapport via un navigateur

scalene profile_forward.py  # Génère un rapport HTML  // Mode forward
python3 -m http.server 8000 # Pour visualiser le rapport via un navigateur

Avec `memory_profiler`

```bash
python3 -m memory_profiler profile_backward.py # Pour le mode backward


Pour avoir un graphe des resumes de memory_profiler
```bash
mprof run profile_backward.py # Pour le mode backward
mprof plot

Pareil pour le mode forward 

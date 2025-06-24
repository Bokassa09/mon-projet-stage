# bobodiff/

Ce dossier contient l’implémentation principale du module de différentiation automatique.

## Contenu des fichiers

- `forward.py` : Implémentation du mode **Forward** (propagation directe).
- `backward.py` : Implémentation du mode **Backward** (rétropropagation).
- `utile.py` : Fonctions utilitaires uniquement pour le mode forward permettant de distinguer les constantes des variables
- `__init__.py` : Rend le dossier importable comme un package Python (`bobodiff`).

## Exemple d'import

```python
# Mode forward
from bobodiff.forward import AutoDiff 
from bobodiff.utile import constante, variable
------------------------------------------
# Mode backward
from bobodiff.backward import Tensor

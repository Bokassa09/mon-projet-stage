
---

### ✅ `examples/README.md`

```markdown
# exemple/

Ce dossier contient des exemples d’utilisation du module de différentiation automatique.

## Fichiers

- `exemple_forward.py` : Exemple avec le mode **Forward**.
- `exemple_backward.py` : Exemple avec le mode **Backward**.

## Fonction testée

On utilise dans les deux cas la fonction suivante :
\[
f(x_1, x_2) = x_1^2 \cdot x_2 + \cos(x_2)
\]

Elle permet de vérifier les calculs de dérivées et gradients manuellement.

## Lancer un exemple

```bash
python exemple/example_forward.py
python exemple/example_backward.py

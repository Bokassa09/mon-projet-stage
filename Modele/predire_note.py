import pandas as pd
from bobodiff.backward import Tensor  

# Chargement des données
data = {
    "Math_Class":     [12, 15, 9, 13, 17, 14, 10, 8, 16, 11, 13, 9, 14, 18, 7, 12, 16, 10, 15, 11],
    "Fr_Class":   [14, 13, 10, 12, 18, 15, 9, 11, 17, 12, 13, 8, 15, 19, 7, 12, 16, 9, 14, 10],
    "Math_Concours":  [11.5, 14, 8, 12, 16, 13, 9, 7.5, 15.5, 10.5, 12.5, 7, 13, 17, 6.5, 11, 15, 8.5, 14.5, 10],
    "Fr_Concours": [13, 12.5, 9, 11.5, 17, 14, 8, 10, 16, 11, 12, 7.5, 14, 18, 6, 11.5, 15, 8, 13.5, 9.5]
}
df = pd.DataFrame(data)

# Normalisation des données 
def normaliser(valeurs):
    mean = sum(valeurs) / len(valeurs)
    return [(v - mean) / 10 for v in valeurs]  # Division par 10 , centré et reduire les données :  x-men(x)/ecart-type ici j'ai pris 10 au lieu de prendre l'ecart type

math_class_norm = normaliser(df["Math_Class"])
fr_class_norm = normaliser(df["Fr_Class"])
math_concours_norm = normaliser(df["Math_Concours"])
fr_concours_norm = normaliser(df["Fr_Concours"])

# Conversion en tensor avec données normalisées
data_tensor = [([Tensor(fr), Tensor(math)], [Tensor(math_conc), Tensor(fr_conc)])
               for fr, math, math_conc, fr_conc in zip(fr_class_norm, math_class_norm, math_concours_norm, fr_concours_norm)]

class LineaireModele:
    def __init__(self):  
        self.w1 = Tensor(0.1)
        self.w2 = Tensor(0.1)
        self.biais = Tensor(0.0)  
    
    def model(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2 + self.biais
   
    def parametre(self):
        return [self.w1, self.w2, self.biais]


model_math = LineaireModele()
model_fr = LineaireModele()

def perte(y_pred, y_vrai):
    diff = y_pred - y_vrai
    return diff.square()

# Entraînement 
learning_rate = 0.01  
tour = 200

print("Début de l'entraînement...")
for i in range(tour):
    total_perte_math = 0
    total_perte_fr = 0
    

    for x, y in data_tensor:
        # Réinitialiser les gradients
        for param in model_math.parametre() + model_fr.parametre():
            param.zero_grad()
        
        x_fr, x_math = x[0], x[1]
        y_math_vrai, y_fr_vrai = y[0], y[1]
        
        # Prédictions
        y_math_pred = model_math.model(x_fr, x_math)
        y_fr_pred = model_fr.model(x_fr, x_math)
        
        # Pertes
        perte_math = perte(y_math_pred, y_math_vrai)
        perte_fr = perte(y_fr_pred, y_fr_vrai)
        
        total_perte_math += perte_math.valeur
        total_perte_fr += perte_fr.valeur
        
        # Rétropropagation
        perte_math.backward()
        perte_fr.backward()
        
        # Mise à jour immédiate des paramètres
        for param in model_math.parametre():
            param.valeur -= learning_rate * param.grad
        for param in model_fr.parametre():
            param.valeur -= learning_rate * param.grad

    # Affichage périodique
    if i % 20 == 0:
        avg_loss_math = total_perte_math / len(data_tensor)
        avg_loss_fr = total_perte_fr / len(data_tensor)
        print(f"Époque {i}: Perte Math={avg_loss_math:.4f}, Perte Fr={avg_loss_fr:.4f}")
        print(f"Poids Math: w1={model_math.w1.valeur:.3f}, w2={model_math.w2.valeur:.3f}, b={model_math.biais.valeur:.3f}")

print("\nEntraînement terminé!")
print(f"Poids finaux Math: w1={model_math.w1.valeur:.3f}, w2={model_math.w2.valeur:.3f}, biais={model_math.biais.valeur:.3f}")
print(f"Poids finaux Fr: w1={model_fr.w1.valeur:.3f}, w2={model_fr.w2.valeur:.3f}, biais={model_fr.biais.valeur:.3f}")

# Test avec dénormalisation
print("\nPrédictions finales (dénormalisées):")
def denormaliser(val_norm, valeurs_originales):
    mean = sum(valeurs_originales) / len(valeurs_originales)
    return val_norm * 10 + mean

for i, (x, y) in enumerate(data_tensor[:5]):  # Afficher seulement les 5 premiers
    x_fr, x_math = x[0], x[1]
    y_math_pred = model_math.model(x_fr, x_math)
    y_fr_pred = model_fr.model(x_fr, x_math)
    
    # Dénormaliser pour l'affichage
    fr_classe_orig = df["Fr_Class"][i]
    math_classe_orig = df["Math_Class"][i]
    math_pred_orig = denormaliser(y_math_pred.valeur, df["Math_Concours"])
    fr_pred_orig = denormaliser(y_fr_pred.valeur, df["Fr_Concours"])
    math_vrai_orig = df["Math_Concours"][i]
    fr_vrai_orig = df["Fr_Concours"][i]
    
    print(f"Élève {i+1}:")
    print(f"  Notes classe: Fr={fr_classe_orig}, Math={math_classe_orig}")
    print(f"  Prédictions: Math={math_pred_orig:.1f}, Fr={fr_pred_orig:.1f}")
    print(f"  Vraies notes: Math={math_vrai_orig}, Fr={fr_vrai_orig}")
    print()
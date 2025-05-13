class AutoDiff:
    " creation la classe AutoDiff "
    def __init__(self, valeur, derive=1):
        self.valeur=valeur
        self.derive=derive
    def __repr__(self):
        return f"AutoDiff : valeur={self.valeur}, deriveé={self.derive}"
    
x=AutoDiff(5.0)
print(x)
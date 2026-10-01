import pandas as pd

clienti = pd.DataFrame({
    "nome": ["Anna", "Luca", "Sara"],
    "eta": [25, 42, 31],
    "spesa": [120.5, 890.0, 340.0],
})

print(clienti)
print("Spesa media:", clienti["spesa"].mean())
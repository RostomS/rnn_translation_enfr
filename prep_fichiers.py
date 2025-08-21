import pandas as pd

donnees = pd.read_csv("/home/rostom/TPnote/archive/en-fr.csv", nrows=50000)

print("Colonnes disponibles :", donnees.columns)

donnees['en'].dropna().to_csv("english.txt", index=False, header=False)
donnees['fr'].dropna().to_csv("french.txt", index=False, header=False)


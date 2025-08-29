# rnn_translation_enfr

## 📌 Contexte
Projet académique réalisé dans le cadre du TP noté **Traduction Automatique** (L3 Intelligence Artificielle, Université Côte d’Azur).  
Objectif : entraîner un modèle RNN encodeur–décodeur pour traduire des phrases de l’anglais vers le français, en explorant différents paramètres d’apprentissage et l’intégration d’un tokenizer **SentencePiece**.

---

## 🛠️ Méthodologie
- Implémentation d’un **RNN caractère-par-caractère** (encoder + decoder).  
- Expériences avec la taille du dataset, du nombre d’époques et des hyperparamètres (`embed_size`, `hidden_size`).  
- Intégration d’un **tokenizer SentencePiece BPE** (16k vocabulaire, tokens spéciaux).  
- Entraînement sur un sous-ensemble du dataset Kaggle (~20 000 paires).  
- Optimiseur : **Adam** avec taux d’apprentissage ajusté (0.001 → 0.0003).  
- Fonction de perte : **CrossEntropyLoss** avec masquage du padding.  

---

## 📊 Résultats
- **Évolution de la perte** : 6.01 → 0.44 (40 époques).  
- **Top-1 Accuracy** ≈ 25.5 %  
- **Top-5 Accuracy** ≈ 26.4 %  
- Observations :  
  - La loss diminue fortement → apprentissage effectif.  
  - Mais les traductions restent souvent incohérentes → limites d’un RNN simple.  
  - L’augmentation de `embed_size` et `hidden_size` améliore les scores (+5 à 10 %), sans dépasser 30 %.  
  - Expérimentation Beam Search → non concluant.  

---

##  Auteur
Projet réalisé par Rostom Samar
🔗 github.com/rostomsamar

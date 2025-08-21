import torch
from torch.utils.data import DataLoader

from model import CharacterRNN
from trainer import Trainer
from dataset import TraductionDataset
from tokenizer import CharTokenizer
import pandas as pd

# On charge les données en augmentant progressivement le nombre de lignes, car la totalité nécessiterait un processeur plus puissant.
jeudedonn= pd.read_csv("/home/rostom/TPnote/archive/en-fr.csv", nrows=5000, header=None)
jeudedonn.columns = ['english', 'french']

# On doit nettoyer nos données pour éviter les erreurs entre les différents types présents dans le fichier CSV
jeudedonn['english'] = jeudedonn['english'].fillna('').astype(str)
jeudedonn['french'] = jeudedonn['french'].fillna('').astype(str)

tokenizer = CharTokenizer()
tokenizer.fit(jeudedonn['english'] + jeudedonn['french'])

# On prépare notre dataset ainsi que notre Dataloader
dataset = TraductionDataset(jeudedonn, tokenizer, max_len=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# On se prépare à configurer notre modèle
taille_vocab = len(tokenizer.char_to_id)
model = CharacterRNN(taille_vocab, embed_size=128, hidden_size=256) # On utilise ces valeurs, dans le but de capturer les relations sans ajouter trop de complexité

# On set l'entraînement
optimizier= torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=2)


trainer = Trainer(model,optimizier, criterion, dataloader, device = 'cuda' if torch.cuda.is_available() else 'cpu' )

#On lance l'entraînement
trainer.train(num_epochs=10)

#On évalue nos métriques
trainer.eval_metriques()

# On traduit une phrase

# J'ajoute la variable device, après avoir rencontré certaines erreurs, on m'indiquait que certains de mes tenseurs, de mon modèle ou de mes entrées sont sur le GPU alors que d'autres sont restés sur le CPU.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def translate(model, tokenizer, sentence, max_len = 100):
    model.eval()
    device = next(model.parameters()).device
    src = tokenizer.encode(sentence)
    src = torch.tensor(src).unsqueeze(0).to(device)

    with torch.no_grad():
        _, hidden = model.encoder(model.embedding(src))

        tgt = [tokenizer.char_to_id['<bos>']]
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt).unsqueeze(0).to(device)
            output, hidden = model.decoder(model.embedding(tgt_tensor), hidden)
            output = model.fc(output[:, -1, :])
            next_token = output.argmax(1).item()
            tgt.append(next_token)

            if next_token == tokenizer.char_to_id['<eos>']:
                break
            if tgt.count(next_token) > 5 :
                break
    return tokenizer.decode(tgt)

print(translate(model, tokenizer,"De La Rue becomes the first person to print"))

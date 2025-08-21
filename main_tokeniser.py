import torch
from torch.utils.data import DataLoader

from model import CharacterRNN
from trainer import Trainer
from dataset import TraductionDataset
from tokenizer import SentencePieceTokenizer
import pandas as pd

# On charge les données en augmentant progressivement le nombre de lignes, car la totalité nécessiterait un processeur plus puissant.
jeudedonn = pd.read_csv("/home/rostom/TPnote/archive/en-fr.csv", nrows=10000, header=None)
jeudedonn.columns = ['english', 'french']

# On nettoie nos données pour éviter les erreurs entre les différents types présents dans le fichier CSV
jeudedonn['english'] = jeudedonn['english'].fillna('').astype(str)
jeudedonn['french'] = jeudedonn['french'].fillna('').astype(str)

# On fait appel au SentencePiece
english_tokenizer = SentencePieceTokenizer("english_v2.model")
french_tokenizer = SentencePieceTokenizer("french_v2.model")

# On prépare notre dataset ainsi que notre Dataloader
dataset = TraductionDataset(jeudedonn, english_tokenizer, french_tokenizer, max_len=100)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# On prépare la configuration de notre modèle
taille_vocab = max(len(english_tokenizer.sp), len(french_tokenizer.sp))  # Calculer la taille max des vocabulaires
model = CharacterRNN(taille_vocab, embed_size=512, hidden_size=1024)  # On utilise ces valeurs pour capturer les relations sans ajouter beaucoup de complexité

# On définit les paramètres de l'entraînement
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


trainer = Trainer(model, optimizer, criterion,  dataloader, device=device)

# On lance l'entraînement
trainer.train(num_epochs=40)

trainer.eval_metriques()


def translate(model, src_tokenizer, tgt_tokenizer, sentence, max_len=100):
    model.eval()
    device = next(model.parameters()).device

    src = src_tokenizer.encode(sentence, out_type=int)
    src = torch.tensor(src + [src_tokenizer.sp.piece_to_id('<eos>')]).unsqueeze(0).to(device)

    with torch.no_grad():
        _, hidden = model.encoder(model.embedding(src))

        tgt = [tgt_tokenizer.sp.piece_to_id('<bos>')]

        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt).unsqueeze(0).to(device)

            output, hidden = model.decoder(model.embedding(tgt_tensor), hidden)
            output = model.fc(output[:, -1, :])
            next_token = output.argmax(1).item()

            if next_token == tgt_tokenizer.sp.piece_to_id('<eos>'):
                break

            tgt.append(next_token)

    return tgt_tokenizer.decode(tgt)

print(translate(model, english_tokenizer, french_tokenizer, "Change the world"))

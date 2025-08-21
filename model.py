import torch.nn as nn


class CharacterRNN(nn.Module):
    def __init__(self ,vocab_size ,embed_size ,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.encoder = nn.RNN(embed_size,hidden_size,batch_first=True)
        self.decoder = nn.RNN(embed_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,vocab_size)

    def forward(self,src, tgt):
        src_embed = self.embedding(src)
        _, hidden = self.encoder(src_embed)

        tgt_embed = self.embedding(tgt)
        outputs, _ = self.decoder(tgt_embed, hidden)
        outputs = self.fc(outputs)
        return outputs



import torch
from tqdm import tqdm



class Trainer:
    def __init__(self, model, optimizer, criterion, dataloader, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device

    def train(self, num_epochs):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()

            for src, tgt in tqdm(self.dataloader, desc=f"Epoch {epoch + 1}"):
                src, tgt = src.to(self.device), tgt.to(self.device)

                # Forward pass
                output = self.model(src, tgt[:, :-1])

                # Calcul de la perte
                loss = self.criterion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
                total_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch + 1} loss: {total_loss/len(self.dataloader):.4f}")


    def eval_metriques(self):
        self.model.eval()
        total_top1 = 0
        total_top5 = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in tqdm(self.dataloader, desc=f"Evaluation des métriques"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_output = tgt[:, 1:]
                outputs = self.model(src, tgt[:, :-1])

                if outputs.size(1) != tgt_output.size(1):
                    long_min = min(outputs.size(1), tgt_output.size(1))
                    outputs = outputs[:, :long_min, :]
                    tgt_output = tgt_output[:, :long_min]

                top5_pred = torch.topk(outputs,5, dim=2).indices
                top1_pred = outputs.argmax(dim=2)

                total_top1 += (top1_pred == tgt_output).sum().item()
                total_top5 += (top5_pred == tgt_output.unsqueeze(-1)).sum().item()
                total_tokens += tgt_output.numel()

        top1_accuracy = (total_top1 / total_tokens) * 100
        top5_accuracy = (total_top5 / total_tokens) * 100
        print(f"Top 1 accuracy: {top1_accuracy:.2f}%")
        print(f"Top 5 accuracy: {top5_accuracy:.2f}%")
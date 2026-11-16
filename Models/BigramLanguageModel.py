import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data.tokenize_data import Tokenizer
from typing import Tuple, Dict

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size : int) -> None:
        super().__init__()
        self.token_embedding_table : nn.Embedding = nn.Embedding(vocab_size, vocab_size)

    @torch.no_grad
    def estimate_loss(self, tokenizer : Tokenizer, eval_iters : int) -> Dict[str, float]:
        out : Dict[str, float] = {}

        self.eval()
        for split in ['train', 'val']:
            losses : torch.Tensor = torch.zeros(eval_iters)

            for k in range(eval_iters):
                X, Y = tokenizer.get_batch(split)

                _, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.train()
        return out
    
    def forward(self, inputs : torch.Tensor, targets : torch.Tensor | None=None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        logits : torch.Tensor = self.token_embedding_table(inputs)

        loss : torch.Tensor | None

        if targets is None:
            loss = None
        else:
            # Reshape for CEL
            B, T, C = logits.shape

            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, inputs : torch.Tensor, max_new_length : int) -> torch.Tensor:
        for _ in range(max_new_length):
            logits, _ = self(inputs, None)

            logits = logits[:, -1, :] # (B, C)

            probabilities : torch.Tensor = F.softmax(logits, dim=-1) # (B, C)

            idx_next : torch.Tensor = torch.multinomial(probabilities, num_samples=1) # (B, 1)

            inputs = torch.cat([inputs, idx_next], dim=1) # (B, T+1)
        return inputs

if __name__ == "__main__":
    tokenizer    : Tokenizer           = Tokenizer(verbose=False)
    bigram_model : BigramLanguageModel = BigramLanguageModel(len(tokenizer.encoder))
    optimizer    : optim.Optimizer     = optim.AdamW(bigram_model.parameters(), lr=0.01)

    batch_size : int = 32

    for steps in range(10_000):
        xb, yb = tokenizer.get_batch('train')

        logits, loss = bigram_model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    loss_report : Dict[str, float] = bigram_model.estimate_loss(tokenizer, 100)

    print(f"Train loss: {loss_report['train']}, validation loss: {loss_report['val']}")

    print(tokenizer.decode(
        bigram_model.generate(
            inputs         = torch.zeros((1, 1), dtype=torch.long), 
            max_new_length = 300
        )[0].tolist()
    ))
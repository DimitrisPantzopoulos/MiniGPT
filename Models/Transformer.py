import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import shutil

from tqdm import tqdm
from typing import Tuple, Dict
from pathlib import Path
from data.tokenize_data import Tokenizer
from training_metrics import TrainingLogger

class AttentionHead(nn.Module):
    def __init__(self, head_size : int, n_embed : int, block_size : int, dropout : float=0.5) -> None:
        super().__init__()

        self.key   : nn.Linear = nn.Linear(n_embed, head_size, bias=False)
        self.query : nn.Linear = nn.Linear(n_embed, head_size, bias=False)
        self.value : nn.Linear = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout : nn.Dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # Input  Shape (Batch, Time, Channels)
        # Output Shape (Batch, Time, Head Size)
        B, T, C = x.shape

        k : torch.Tensor = self.key(x)   # (Batch, Time, Head Size)
        q : torch.Tensor = self.query(x) # (Batch, Time, Head Size)

        # Compute Attention Scores: How much should token pay attention to each other token.
        weights : torch.Tensor = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (Batch, Time, Head Size) @ (B, Head Size, T) -> (Batch, Time, Time) # type: ignore
        weights = F.softmax(weights, dim=-1) # (Batch, Time, Time)
        weights = self.dropout(weights)

        v : torch.Tensor = self.value(x)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            num_heads  : int, 
            n_embed    : int, 
            block_size : int,
            head_size  : int,
            dropout    : float = 0.5,
        ) -> None:
        super().__init__()

        self.heads      : nn.ModuleList = nn.ModuleList([AttentionHead(head_size, n_embed, block_size, dropout) for _ in range(num_heads)])
        self.projection : nn.Linear     = nn.Linear(head_size * num_heads, n_embed)
        self.dropout    : nn.Dropout    = nn.Dropout(dropout)


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        out : torch.Tensor = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(out))

class FeedForward(nn.Module):
    def __init__(self, n_embed : int) -> None:
        super().__init__()

        self.linear : nn.Linear = nn.Linear(n_embed, n_embed)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(x))

class Block(nn.Module):
    def __init__(
            self, 
            n_embed    : int, 
            n_head     : int,
            block_size : int,
        ) -> None:
        super().__init__()

        head_size : int = n_embed // n_head

        self.self_attention : MultiHeadAttention = MultiHeadAttention(n_head, n_embed, block_size, head_size)
        self.feed_forward   : FeedForward  = FeedForward(n_embed)
        self.layer_norm_1   : nn.LayerNorm = nn.LayerNorm(n_embed)
        self.layer_norm_2   : nn.LayerNorm = nn.LayerNorm(n_embed)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x

class GPTModel(nn.Module):
    def __init__(
            self,
            vocab_size : int = 65,
            n_embed    : int = 384,
            block_size : int = 256,
            n_head     : int = 6,
            n_layer    : int = 6,
            device     : torch.device = torch.device('cpu')
        ) -> None:
        super().__init__()

        self.block_size : int = block_size
        self.device : torch.device = device
        self.model_config = dict(
            vocab_size=vocab_size, n_embed=n_embed, block_size=block_size,
            n_head=n_head, n_layer=n_layer,
        )

        self.token_embedding_table    : nn.Embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table : nn.Embedding = nn.Embedding(block_size, n_embed)

        self.blocks     : nn.Sequential = nn.Sequential(*[Block(n_embed, n_head, block_size) for _ in range(n_layer)])
        self.layer_norm : nn.LayerNorm  = nn.LayerNorm(n_embed)
        self.output_ln  : nn.Linear     = nn.Linear(n_embed, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad
    def estimate_loss(self, tokenizer : Tokenizer, eval_iters : int) -> Dict[str, float]:
        out : Dict[str, float] = {}

        self.eval()
        for split in ['train', 'val']:
            losses : torch.Tensor = torch.zeros(eval_iters)

            for k in range(eval_iters):
                X, Y = tokenizer.get_batch(split)

                X, Y = X.to(self.device), Y.to(self.device)

                _, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.train()
        return out

    def forward(self, inputs : torch.Tensor, targets : torch.Tensor | None=None) -> Tuple[torch.Tensor, torch.Tensor | None]:
        B, T = inputs.shape

        token_embeddings    : torch.Tensor = self.token_embedding_table(inputs)
        position_embeddings : torch.Tensor = self.position_embedding_table(torch.arange(T, device=self.device))

        x : torch.Tensor = token_embeddings + position_embeddings
        x = self.blocks(x)
        x = self.layer_norm(x)

        logits : torch.Tensor = self.output_ln(x)

        loss : torch.Tensor | None = None

        if targets is not None:
            B, T, C = logits.shape

            logits  = logits.view( B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x : torch.Tensor, max_new_tokens : int=1_000) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Clip inputs to the last block size, so embeddings work properly
            x_cond : torch.Tensor = x[:, -self.block_size:]

            logits, _ = self(x_cond)

            logits = logits[:, -1, :] # (B, C)

            probabilities : torch.Tensor = F.softmax(logits, dim=-1) # (B, C)

            next_idx : torch.Tensor = torch.multinomial(probabilities, num_samples=1) # (B, 1)

            x = torch.cat([x, next_idx], dim=1) # (B, T+1)
        return x

    @torch.no_grad()
    def generate_text(model, tokenizer, max_new_tokens=500):
        model.eval()

        # Start with token 0
        context = torch.zeros(
            (1, 1),
            dtype=torch.long,
            device=next(model.parameters()).device,
        )

        generated = model.generate(context, max_new_tokens=max_new_tokens)
        text = tokenizer.decode(generated[0].tolist())

        print(text)
        return text

    def save_checkpoint(
            self,
            path: str | Path,
            tokenizer: Tokenizer,
            step: int,
            losses: Dict[str, float],
            optimizer: optim.Optimizer,
        ) -> None:
        """Save weights, model/tokenizer configuration and optimizer state atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + '.tmp')
        checkpoint = {
            'model_state_dict': {key: value.detach().cpu() for key, value in self.state_dict().items()},
            'model_config': self.model_config,
            'chars': sorted(tokenizer.encoder, key=tokenizer.encoder.get),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'losses': losses,
        }
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def train_model(
            self,
            tokenizer : Tokenizer,
            max_iters : int = 5000,
            *,
            eval_interval : int = 50,
            eval_iters    : int = 100,
            log_dir       : str | Path | None = None,
            resume_from   : str | Path | None = None,
        ) -> None:
        if min(max_iters, eval_interval, eval_iters) < 1:
            raise ValueError("max_iters, eval_interval and eval_iters must be positive.")

        # Print Model Params
        print(sum(p.numel() for p in self.parameters()) / 1e6, 'M Parameters')

        self.to(self.device)
        self.train()

        optimizer : optim.Optimizer = optim.AdamW(self.parameters(), lr=3e-4)
        start_step = 0
        best_val_loss = float('inf')
        best_source = None
        if resume_from is not None:
            resume_path = Path(resume_from)
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=True)
            if checkpoint['model_config'] != self.model_config:
                raise ValueError('Checkpoint architecture does not match this model.')
            chars = sorted(tokenizer.encoder, key=tokenizer.encoder.get)
            if checkpoint['chars'] != chars:
                raise ValueError('Checkpoint vocabulary does not match this dataset.')
            start_step = checkpoint['step']
            if max_iters <= start_step:
                raise ValueError(f'max_iters must be greater than checkpoint step {start_step}.')
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['losses']['val']
            best_source = resume_path
            previous_best_path = resume_path.with_name('best.pt')
            if previous_best_path.is_file():
                previous_best = torch.load(previous_best_path, map_location='cpu', weights_only=True)
                if (previous_best['model_config'] == self.model_config and previous_best['chars'] == chars
                        and previous_best['losses']['val'] < best_val_loss):
                    best_val_loss = previous_best['losses']['val']
                    best_source = previous_best_path
            print(f'Resuming from step {start_step}; target step {max_iters}.')

        with TrainingLogger(log_dir) as logger:
            print(f"Training results: {logger.run_dir.resolve()}")
            print(f"Checkpoints: {logger.run_dir.resolve() / 'best.pt'} and latest.pt")
            if best_source is not None:
                shutil.copyfile(best_source, logger.run_dir / 'best.pt')

            def record_loss(step: int) -> None:
                nonlocal best_val_loss
                losses = self.estimate_loss(tokenizer, eval_iters)
                self.save_checkpoint(logger.run_dir / 'latest.pt', tokenizer, step, losses, optimizer)
                if losses['val'] < best_val_loss:
                    self.save_checkpoint(logger.run_dir / 'best.pt', tokenizer, step, losses, optimizer)
                    best_val_loss = losses['val']
                logger.log(step, losses)
                tqdm.write(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            record_loss(start_step)
            for step in tqdm(range(start_step + 1, max_iters + 1), initial=start_step,
                             total=max_iters, desc='Training...', unit='iteration'):
                x_batch, y_batch = tokenizer.get_batch('train')
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                _, loss = self(x_batch, y_batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if step % eval_interval == 0 or step == max_iters:
                    record_loss(step)
                




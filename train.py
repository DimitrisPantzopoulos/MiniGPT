import argparse
from pathlib import Path

import torch

from Models.Transformer import GPTModel
from data.tokenize_data import Tokenizer


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train MiniGPT and save TensorBoard logs, CSV losses and a chart.")
    parser.add_argument("--max-iters", type=positive_int, default=5000)
    parser.add_argument("--eval-interval", type=positive_int, default=50)
    parser.add_argument("--eval-iters", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--block-size", type=positive_int, default=256)
    parser.add_argument("--n-embed", type=positive_int, default=384)
    parser.add_argument("--n-head", type=positive_int, default=6)
    parser.add_argument("--n-layer", type=positive_int, default=6)
    parser.add_argument("--data-path", type=Path, default=root / "data" / "raw" / "input.txt")
    parser.add_argument("--log-dir", type=Path, help="New directory for this run; defaults to runs/<timestamp>.")
    parser.add_argument("--resume", type=Path, help="Restore weights, optimizer and completed step from a checkpoint.")
    args = parser.parse_args()
    if args.n_embed % args.n_head:
        parser.error("--n-embed must be divisible by --n-head")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoint_config = None
    if args.resume is not None:
        checkpoint_config = torch.load(args.resume, map_location="cpu", weights_only=True)["model_config"]
    tokenizer = Tokenizer(
        path=str(args.data_path),
        block_size=checkpoint_config["block_size"] if checkpoint_config else args.block_size,
        batch_size=args.batch_size,
        verbose=False,
    )
    model_config = checkpoint_config or dict(
        vocab_size=len(tokenizer.encoder), block_size=args.block_size,
        n_embed=args.n_embed, n_head=args.n_head, n_layer=args.n_layer,
    )
    model = GPTModel(**model_config, device=device)
    model.train_model(
        tokenizer,
        max_iters     = args.max_iters,
        eval_interval = args.eval_interval,
        eval_iters    = args.eval_iters,
        log_dir       = args.log_dir,
        resume_from   = args.resume,
    )

if __name__ == "__main__":
    main()


"""Generate text from a downloaded checkpoint without the training dataset."""

import argparse
from pathlib import Path

import torch

from Models.Transformer import GPTModel


@torch.no_grad()
def generate_text(checkpoint_path: str | Path, prompt: str = "", max_new_tokens: int = 500) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = GPTModel(**checkpoint["model_config"], device=device).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    chars = checkpoint["chars"]
    encoder = {char: index for index, char in enumerate(chars)}
    unknown = set(prompt) - set(encoder)
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {sorted(unknown)!r}")
    tokens = [encoder[char] for char in prompt] if prompt else [0]
    context = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)
    return "".join(chars[index] for index in generated[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be nonnegative")
    try:
        text = generate_text(args.checkpoint, args.prompt, args.max_new_tokens)
    except ValueError as error:
        parser.error(str(error))
    print(text)


if __name__ == "__main__":
    main()

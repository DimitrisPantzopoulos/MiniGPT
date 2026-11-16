from typing import List, Dict, Tuple
from pathlib import Path

import torch

class Tokenizer:
    def __init__(
            self, 
            path            : str | None = None,
            block_size      : int   = 8,
            batch_size      : int   = 4,
            train_val_split : float = 0.9,
            verbose         : bool  = True
        ) -> None:
        # General Tokenizer Information
        self.dataset_path    : Path  = Path.cwd() / 'data' / 'raw' / 'input.txt' if path is None else Path(path)
        self.train_val_split : float = train_val_split
        self.block_size      : int   = block_size
        self.batch_size      : int   = batch_size
        self.verbose         : bool  = verbose

        # Make the encoder/decoder
        self.encoder : Dict[str, int]
        self.decoder : Dict[int, str]

        self.intialise_coders()

        # Encode the Corpus
        self.encoded_corpus : torch.Tensor

        self.encode_corpus()

        # Make the test and train dataset
        self.train_dataset : torch.Tensor
        self.test_dataset  : torch.Tensor

        self.make_train_val_datasets()

    def intialise_coders(self) -> None:
        # Open the dataset file and read all contents
        with open(self.dataset_path, mode='r', encoding='utf-8') as file:
            dataset : str = file.read()

            all_chars  : List[str] = sorted(list(set(dataset)))
            vocab_size : int = len(all_chars)

            if self.verbose:
                print(f"All Characters: {all_chars}")
                print(f"Vocab Size: {vocab_size}")

            self.encoder = { char : i for i, char in enumerate(all_chars) }
            self.decoder = { i : char for i, char in enumerate(all_chars) }

    def encode_corpus(self) -> None:
        with open(self.dataset_path, mode='r', encoding='utf-8') as file:
            dataset : str = file.read()

            self.encoded_corpus = torch.tensor(self.encode(dataset), dtype=torch.long)

            if self.verbose:
                # Print Dataset Stats
                print("Encoded Dataset")
                print(f"Shape: {self.encoded_corpus.shape}, Type: {self.encoded_corpus.dtype}")
                print(f"First 1K tokens: \n{self.encoded_corpus[:1000]}")

    def make_train_val_datasets(self) -> None:
        train_n : int = int(self.train_val_split * len(self.encoded_corpus))

        self.train_dataset = self.encoded_corpus[:train_n]
        self.test_dataset  = self.encoded_corpus[train_n:]

    def get_batch(self, split : str) -> Tuple[torch.Tensor, torch.Tensor]:
        data : torch.Tensor = self.train_dataset if split == 'train' else self.test_dataset

        ix : torch.Tensor = torch.randint(len(data) - self.block_size, (self.batch_size, ))
        x  : torch.Tensor = torch.stack([data[i : i + self.block_size] for i in ix])
        y  : torch.Tensor = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y
    
    def encode(self, phrase : str) -> List[int]:
        return [self.encoder[char] for char in phrase]

    def decode(self, encoded_phrase : List[int]) -> str:
        return "".join([self.decoder[token] for token in encoded_phrase])
    
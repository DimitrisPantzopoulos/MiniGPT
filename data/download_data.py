from urllib.request import urlopen
from pathlib import Path
from typing import List
from tqdm import tqdm

DATASET_TIMEOUT : int = 60

DATASET_FILES : List[str] = ['input.txt']
DATASET_URL   : str = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/'

def download_data() -> None:
    raw_dir : Path = Path.cwd() / 'data' / 'raw'
    raw_dir.mkdir(exist_ok=True)

    for filename in tqdm(DATASET_FILES, desc='Downloading Dataset.', unit='files'):
        with urlopen(DATASET_URL + filename, timeout=DATASET_TIMEOUT) as response:
            data = response.read()

        file_destination : Path = raw_dir / filename

        temporary_file : Path = file_destination.with_suffix('.csv.temp')
        temporary_file.write_bytes(data)
        temporary_file.replace(file_destination)

        print(f"Saved {len(DATASET_FILES)} files to {raw_dir}")

if __name__ == "__main__":
    download_data()
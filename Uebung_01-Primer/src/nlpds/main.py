from pathlib import Path
from typing import Collection, Counter

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from nlpds.abc.ex1.primer import BiGram
from nlpds.submission.ex1.primer import (
    BiGramGenerator,
    BinaryLanguageClassifier,
    LanguageClassificationDataset,
)

if __name__ == "__main__":
    """
    Training script template. Change as you see fit!
    """
    ## Setup: Data
    data_root = Path("../data/ex1/primer/")
    deu_dev = data_root / "deu_dev.txt"
    deu_test = data_root / "deu_test.txt"
    deu_train = data_root / "deu_train.txt"
    eng_dev = data_root / "eng_dev.txt"
    eng_test = data_root / "eng_test.txt"
    eng_train = data_root / "eng_train.txt"

    def double_lower_BiGram() -> Counter:
        """
        Creates a Counter with lowercase Bigrams as keys and zeros as values.
        """
        i = 97
        j = 97
        vocab = []
        while i < 123:
            while j < 123:
                vocab.append(chr(i) + chr(j))
                j += 1
            j = 97
            i += 1
        return Counter(dict.fromkeys(vocab, 0))


    vocabulary: Collection[BiGram] = double_lower_BiGram()
    train_dataset: LanguageClassificationDataset = ...  # TODO
    dev_dataset: LanguageClassificationDataset = ...  # TODO
    test_dataset: LanguageClassificationDataset = ...  # TODO

    ## Setup: Training Hyper-Parameters
    model: BinaryLanguageClassifier = ...  # TODO

    criterion = ...  # TODO

    learning_rate: float = ...  # TODO
    optimizer: torch.optim.Optimizer = ...  # TODO

    num_epochs: int = ...  # TODO
    batch_size: int = ...  # TODO

    ## Training
    dev_dataloader: DataLoader = ...  # TODO
    for epoch in trange(num_epochs, desc="Epoch"):
        train_dataloader: DataLoader = ...  # TODO

        model.train()
        for batch in train_dataloader:
            ...  # TODO

        ## Evaluation: Dev Set
        model.eval()
        with torch.no_grad():
            for batch in dev_dataloader:
                ...  # TODO

    ## Evaluation: Test Set
    test_dataloader: DataLoader = ...  # TODO

    model.eval()
    with torch.no_grad():
        ...  # TODO

    ## Evaluation: Save Results?
    ...  # TODO

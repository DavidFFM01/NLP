from pathlib import Path
from typing import Collection, Counter, Self

import torch
from torch import Tensor

from nlpds.abc.ex1.primer import (
    AbstractBiGramGenerator,
    AbstractBinaryLanguageClassifier,
    AbstractLanguageClassificationDataset,
    BiGram,
)


class BinaryLanguageClassifier(AbstractBinaryLanguageClassifier):
    def __init__(
        self,
        # ...
    ):
        # TODO: Implement the constructor
        raise NotImplementedError

    # TODO: Implement all methods from AbstractBinaryLanguageClassifier
    # TODO: Document all methods from AbstractBinaryLanguageClassifier

    @property
    def num_features(self) -> int:
        """
        Returns the size of the input features.
        """
        raise NotImplementedError

    def forward(self, features: Tensor) -> Tensor:
        """
        Forward pass of the classifier.
        Returns unnormalized logits for binary classification.
        """
        raise NotImplementedError

    @property
    def weights(self) -> Tensor:
        """
        Gets the weights of the classifier.
        """
        raise NotImplementedError

    @weights.setter
    def weights(self, weights: Tensor):
        """
        Sets the weights of the classifier.
        """
        raise NotImplementedError

    @property
    def bias(self) -> Tensor:
        """
        Gets the bias of the classifier.
        """
        raise NotImplementedError

    @bias.setter
    def bias(self, bias: Tensor):
        """
        Sets the bias of the classifier.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """
        Get the current device.
        """
        raise NotImplementedError

    @device.setter
    def device(self, device: str | torch.device):
        """
        Move all PyTorch modules to a given device.
        """
        raise NotImplementedError

    def to(self, device: str | torch.device) -> Self:
        """
        Moves the classifier to a given device.
        """
        super().to(device)
        self.device = torch.device(device)
        return self

    @classmethod
    def with_num_features(cls, num_features: int) -> Self:
        """
        Create a binary language classifier with a given number of features.
        The classifier is expected to have randomly initialized weights and a bias.
        """
        raise NotImplementedError


class LanguageClassificationDataset(AbstractLanguageClassificationDataset):
    def __init__(
        self,
        file_de: Path,
        file_en: Path,
        file_len: int
    ) -> None:
        self.file_de = file_de
        self.file_en = file_en

    def __len__(self) -> int:
        return len(self.file_en)
    
    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return super().__getitem__(index)

    def from_files(
        cls,
        file_de: Path,
        file_en: Path,
        vocabulary: Collection[BiGram],
    ) -> Self:
    	raise NotImplementedError


class BiGramGenerator(AbstractBiGramGenerator):
    def __init__(
        self,
        vocabulary: Collection[BiGram]
    ):
        self.vocabulary = vocabulary
        """
        Vocabulary (list) containing valid bigrams
        """

    def bi_grams(self, sentence: str) -> list[BiGram]:
        """
        Generate bi-grams from a sentence.
        Returns a list of bi-grams (str's of length 2).
        If the sentence contains no valid bi-grams, returns an empty list.
        """
        string = sentence.lower()
        list_of_bigrams = []
        for i in range(len(string)-1):
                if self.vocabulary.__contains__(string[i:i+2]):
                    list_of_bigrams.append(string)
        return list_of_bigrams
                     

    def forward(self, sentence: str) -> Tensor:
        """
        Generate a bi-gram-frequency feature vector for a sentence.
        """
        list_bigrams = self.bi_grams(sentence)
        counter_bigrams = Counter(list_bigrams)
        return torch.tensor(counter_bigrams)

    def __call__(self, sentence: str) -> Tensor:
        return self.forward(sentence)

    def __getitem__(self, bi_gram: BiGram) -> int:
        """
        Returns the index of a bi-gram.
        If the given bi-gram is not in the vocabulary, raises a KeyError.
        If the given value is not a valid bi-gram, raises a ValueError.
        """
        raise NotImplementedError

    def get(self, bi_gram: BiGram, default=-1) -> int:
        """
        Returns the index of a bi-gram.
        If the given value is not a valid bi-gram or not in the vocabulary,
        returns the default value.
        """
        return self[bi_gram] if bi_gram in self else default

    def __len__(self) -> int:
        """
        Return the number of bi-grams in the vocabulary.
        """
        return len(self.vocabulary)

    def __contains__(self, value: BiGram) -> bool:
        """
        Check if a bi-gram is in the vocabulary.
        If the given value is not a valid bi-gram, return False.
        """
        return self.vocabulary.__contains__(value)

    @property
    def vocabulary(self) -> Collection[BiGram]:
        """
        Get the vocabulary of bi-grams.
        """
        return self.vocabulary

    @classmethod
    def from_vocabulary(cls, vocabulary: Collection[BiGram]) -> Self:
        """
        Create a bi-gram generator from a vocabulary of valid bi-grams.
        """
        bigrams = Collection[BiGram]
        return cls(bigrams)
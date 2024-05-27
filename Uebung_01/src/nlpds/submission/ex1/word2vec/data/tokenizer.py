from altair import Self

from nlpds.abc.ex1.word2vec.data.tokenizer import (
    PreTokenizerABC,
    TokenizedSentenceABC,
    TokenizerABC,
)


class TokenizedSentence(TokenizedSentenceABC):
    def __init__(
        self,
        # ...
    ):
        # TODO: Implement the constructor
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    # TODO: Implement all methods from TokenizedSentenceABC
    # TODO: Document all methods from TokenizedSentenceABC


class PreTokenizer(PreTokenizerABC):
    def __init__(
        self,
        # ...
    ):
        # TODO: Implement the constructor
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    # TODO: Implement all methods from PreTokenizerABC
    # TODO: Document all methods from PreTokenizerABC


class Tokenizer(TokenizerABC):
    def __init__(
        self,
        # ...
    ):
        # TODO: Implement the constructor
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    # TODO: Implement all methods from TokenizerABC
    # TODO: Document all methods from TokenizerABC

    class Fitter(TokenizerABC.Fitter[Self]):
        def __init__(
            self,
            # ...
        ):
            # TODO: Implement the constructor
            raise NotImplementedError(
                f"{self.__class__.__name__}.__init__() is not implemented"
            )

        # TODO: Implement all methods from TokenizerABC.Fitter
        # TODO: Document all methods from TokenizerABC.Fitter



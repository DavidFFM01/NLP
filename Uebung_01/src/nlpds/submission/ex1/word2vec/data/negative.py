from nlpds.abc.ex1.word2vec.data.negative import (
    DatasetNegativeSamplingABC,
    NegativeSample,
    NegativeSamplerABC,
)


class NegativeSampler(NegativeSamplerABC):
    def __init__(
        self,
        # ...
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    # TODO: Implement all methods from NegativeSamplerABC
    # TODO: Document all methods from NegativeSamplerABC


class DatasetNegativeSampling(DatasetNegativeSamplingABC[NegativeSample]):
    def __init__(
        self,
        # ...
    ) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__}.__init__() is not implemented"
        )

    # TODO: Implement all methods from DatasetNegativeSamplingABC
    # TODO: Document all methods from DatasetNegativeSamplingABC

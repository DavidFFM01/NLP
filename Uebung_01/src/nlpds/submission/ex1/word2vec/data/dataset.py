from nlpds.abc.ex1.word2vec.data.dataset import DatasetABC, Word2VecSample


class Dataset(DatasetABC[Word2VecSample]):
    def __init__(
        self,
        # ...
    ):
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented")

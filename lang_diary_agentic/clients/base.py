import abc
import typing as ty


class ClientEmbeddingModel(abc.ABC):
    pass


class ClientLLM(abc.ABC):
    @abc.abstractmethod
    def get_available_models(self) -> ty.List[str]:
        raise NotImplementedError()

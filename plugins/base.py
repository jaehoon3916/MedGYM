from abc import ABC, abstractmethod
from typing import Any


class BasePlugin(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

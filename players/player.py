from abc import ABC, abstractmethod
from shapely import Point

from src.cake import Cake


class PlayerException(Exception):
    pass


class Player(ABC):
    def __init__(self, children: int, cake: Cake, cake_path: str | None) -> None:
        self.children = children
        self.cake = cake
        self.cake_path = cake_path

    def __str__(self) -> str:
        return f"{self.__module__}()"

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def get_cuts(self) -> list[tuple[Point, Point]]:
        pass

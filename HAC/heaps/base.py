from HAC.tools.nodes import BaseNode as Node
from typing import Any, Union, Generator
from abc import ABC, abstractmethod


class BaseHeap(ABC):

    def __init__(self) -> "BaseHeap":
        self._max_node: Node = None
        self._total_nodes: int = 0

    @abstractmethod
    def push(
        self,
        value: Union[int, float],
        key: Any
        ) -> Node:
        pass

    @property
    def find_max(self) -> Node:
        return self._max_node

    @property
    def total_nodes(self) -> int:
        return self._total_nodes
    
    @abstractmethod
    def increase_value(
        self, 
        node: Node, 
        new_value: Union[int, float]
        ) -> None:
        pass

    @abstractmethod
    def merge(
        self, 
        heap: "BaseHeap"
        ) -> None:
        pass

    @abstractmethod
    def pop(self) -> Node:
        pass

    @abstractmethod
    def delete_node(
        self, 
        node: Node
        ) -> None:
        pass

    @abstractmethod
    def iterate(self) -> Generator[Node, None, None]:
        pass

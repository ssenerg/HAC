from typing import Any, Union


class BaseNode:

    """
    Base Node
    ------------------

    This Node class is created to store data of heaps

    Params:
        value (int OR float): The value of the node
        key (Any): The key of the node
    """

    def __init__(
            self, 
            value: Union[int, float], 
            key: Any) -> "BaseNode":
        
        self._value: Union[int, float] = value
        self._key: Any = key
    
    def __str__(self) -> str:
        return f'Key: {self._key} Value: {self._value}'
    
    @property
    def key(self):
        return self._key
    
    @property
    def value(self):
        return self._value


class DoublyLinkedNode(BaseNode):

    """
    Doubly Linked Node
    ------------------

    This Node class is created to store doubly linked list
    specialized for Max Fibonacci Heap

    Params:
        value (int OR float): The value of the node
        key (Any): The key of the node
    """

    def __init__(
            self, 
            value: Union[int, float], 
            key: Any
        ) -> "DoublyLinkedNode":

        super().__init__(value, key)
        self._parent: Union[DoublyLinkedNode, None] = None
        self._child: Union[DoublyLinkedNode, None] = None
        self._left: Union[DoublyLinkedNode, None] = None
        self._right: Union[DoublyLinkedNode, None] = None
        self._degree: int = 0
        self._loser: bool = False

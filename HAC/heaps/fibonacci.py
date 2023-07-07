from HAC.tools.nodes import DoublyLinkedNode as Node
from typing import Any, Union, Generator
from HAC.heaps.base import BaseHeap
from math import log


class MaxFibonacciHeap(BaseHeap):

    """
    Max Fibonacci Heap
    ------------------
    A max fibonacci heap is a data structure that supports the following
    operations in O(1) amortized time:
        - Push
        - Find Max
        - Increase Value
        - Merge

    and the following operations in O(log(n)) amortized time:
        - Pop
        - Delete Node
    """

    def __init__(self) -> "MaxFibonacciHeap":

        self.__root_list: Node = None # Pointer to the first node of root
        self.__max_node: Node = None # Pointer to the head and maximum node
        self.__total_nodes: int = 0 # Total node count in the Heap

    def push(
            self, 
            value: Union[int, float], 
            key: Any = None
        ) -> Node:

        """
        Push a new node into the heap in O(1)

        ------------------

        Params:
            value (int OR float): The value of the node
            key (Any): The key of the node
        Returns:
            The node that was pushed into the heap
        """

        node = Node(value, key)
        node._left = node._right = node
        self.__merge_with_root_list(node)

        if self.__max_node is None or node._value > self.__max_node._value:
            self.__max_node = node

        self.__total_nodes += 1

        return node
    
    def increase_value(
            self, 
            node: Node, 
            new_value: Union[int, float]
        ) -> None:

        """
        Increase the value of a node in the heap in O(1)
        
        ------------------
        
        Params:
            node (Node): The node to increase the value of
            new_value (int OR float): The new value of the node
        """

        if new_value <= node._value:
            return None
        
        node._value, node_parent = new_value, node._parent

        if node_parent is not None and node._value > node_parent._value:
            self.__cut(node, node_parent)
            self.__cascading_cut(node_parent)

        # Update max node if needed
        if node._value > self.__max_node._value:
            self.__max_node = node

    def merge(
            self, 
            heap: "MaxFibonacciHeap"
        ) -> None:

        """
        Merge another heap to this heap in O(1)

        ------------------

        Params:
            heap (MaxFibonacciHeap): The heap to merge with this heap
        """

        # Fix pointers when merging the two heaps
        last = heap.__root_list._left
        heap.__root_list._left = self.__root_list._left
        self.__root_list._left._right = heap.__root_list
        self.__root_list._left = last
        self.__root_list._left._right = self.__root_list

        # Update max node if needed
        if heap.__max_node._value > self.__max_node._value:
            self.__max_node = heap.__max_node

        # Update total nodes
        self.__total_nodes += heap.__total_nodes

    def pop(self) -> Union[Node, None]:

        """
        Pop the max node from the heap in O(log(n))

        ------------------

        Returns:
            The max node in the heap OR None if heap is empty
        """
        
        max_node = self.__max_node
        if max_node is None:
            return
        
        if max_node._child is not None:
            # Attach child nodes to root list
            children = [_ for _ in self.__class__.__iterate(max_node._child)]
            for child in children:
                self.__merge_with_root_list(child)

        self.__remove_from_root_list(max_node)

        # Set new max node in heap
        if max_node == max_node._right:
            self.__max_node = self.__root_list = None
        else:
            self.__max_node = max_node._right
            self.__consolidate()

        self.__total_nodes -= 1

        return max_node

    def delete_node(
            self, 
            node: Node
        ) -> None:
        
        """
        Delete specific node from the heap

        ------------------

        Params:
            node (Node): The node to delete from the heap
        """

        self.increase_value(node, self.__max_node._value + 1)
        self.pop()

    def iterate(self) -> Generator[Node, None, None]:

        """
        Iterate over whole heap

        ------------------

        Returns:
            Node
        """

        def __iterate(node):
            for node_ in MaxFibonacciHeap.__iterate(node):
                yield node_
                if node_._child is not None:
                    yield from __iterate(node_._child)

        yield from __iterate(self.__root_list)

    @staticmethod
    def __iterate(head: Node) -> Generator[Node, None, None]:

        """
        Iterate through a doubly linked list

        ------------------

        Params:
            head (Node): The head of the doubly linked list
        Yields:
            Node: The next node in the doubly linked list
        """
        
        node = stop = head
        while True:
            yield node
            node = node._right
            if node == stop:
                break

    def __cut(
            self, 
            child: Node, 
            parent: Node
        ) -> None:

        """
        Cut a node from its parent node and bring it up to the root list
        
        ------------------
        
        Params:
            child (Node): The node to cut from its parent
            parent (Node): The parent node to cut the child from
        """

        MaxFibonacciHeap.__remove_from_child_list(parent, child)
        self.__merge_with_root_list(child)
        child._loser = False

    def __cascading_cut(
            self, 
            parent: Node
        ) -> None:

        """
        Cascading cut of parent node to obtain good time bounds
        
        ------------------
        
        Params:
            parent (Node): The parent node to perform cascading cut on
        """
        
        parent_parent = parent._parent
        
        if parent_parent is not None:
            if parent._loser is False:
                parent._loser = True
            else:
                self.__cut(parent, parent_parent)
                self.__cascading_cut(parent_parent)

    def __consolidate(self) -> None:

        """
        Combine root nodes of equal degree to consolidate the heap
        """

        store = [None] * int(log(self.__total_nodes, (1 + 5 ** .5)/2) + 1)
        nodes = [_ for _ in self.__class__.__iterate(self.__root_list)]

        for node in nodes:

            degree = node._degree
            
            while store[degree] is not None:

                same_degree_node = store[degree]
                if node._value < same_degree_node._value:
                    node, same_degree_node = same_degree_node, node

                self.__heap_link(same_degree_node, node)

                store[degree] = None
                degree += 1
            
            store[degree] = node
        
        # Find new max node
        for node in store:
            if node is None:
                continue
            if node._value > self.__max_node._value:
                self.__max_node = node

    def __heap_link(
            self, 
            child: Node, 
            parent: Node
        ) -> None:

        """
        Link a node under another node in the root list

        ------------------

        Params:
            child (Node): The node to link under another node
            parent (Node): The node to link another node under
        """

        self.__remove_from_root_list(child)

        child._left = child._right = child
        MaxFibonacciHeap.__merge_with_child_list(parent, child)

    def __merge_with_root_list(
            self, 
            node: Node
        ) -> None:

        """
        Merge a node with the doubly linked root list

        ------------------

        Params:
            node (Node): The node to merge with the root list
        """
        
        node._parent = None

        if self.__root_list is None:
            self.__root_list = node
        else:
            node._right = self.__root_list._right
            node._left = self.__root_list
            self.__root_list._right._left = node
            self.__root_list._right = node

    @staticmethod
    def __merge_with_child_list(
            parent, 
            child
        ) -> None:

        """
        Merge a node with the doubly linked child list of a root node

        ------------------

        Params:
            parent (Node): The root node to merge a node with
            child (Node): The node to merge with the child list of parent
        """
        
        if parent._child is None:
            parent._child = child
        else:
            child._right = parent._child._right
            child._left = parent._child
            parent._child._right._left = child
            parent._child._right = child
        
        parent._degree += 1
        child._parent = parent
        child._loser = False

    def __remove_from_root_list(
            self, 
            node: Node
        ) -> None:

        """
        Remove a node from the doubly linked root list
        
        ------------------
        
        Params:
            node (Node): The node to remove from the root list
        """

        if node == self.__root_list:
            self.__root_list = node._right
        node._left._right = node._right
        node._right._left = node._left

    @staticmethod
    def __remove_from_child_list(parent: Node, child: Node) -> Node:

        """
        Remove a node from the doubly linked child list
        
        ------------------
        
        Params:
            parent (Node): The root node of the child list
            child (Node): The node to remove from the child list
        """

        if parent._child == parent._child._right:
            parent._child = None
        elif parent._child == child:
            parent._child = child._right
            child._right._parent = parent

        child._left._right = child._right
        child._right._left = child._left
        parent._degree -= 1

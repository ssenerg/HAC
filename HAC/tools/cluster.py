from typing import Union, List


class Cluster:

    """
    Cluster
    ------------------

    This class is created to store data of clusters

    Params:
        value (int): The value of the node
    """

    def __init__(
            self, 
            value: int = None
        ) -> "Cluster":

        self.activate: bool = True
        self.left: Union["Cluster", int, None] = value
        self.right: Union["Cluster", int, None] = None

        if value is not None:
            self.values: List[int] = [value]
        else:
            self.values: List[int] = []

    def __str__(self):
        return f"({self.left},{self.right})"

    def merge(self, cluster: "Cluster") -> None:

        """
        This function is used to merge two clusters
        
        ------------------

        Params:
            cluster (Cluster): The cluster to be merged
        """

        cluster.activate = False

        if self.right is None:
            if cluster.right is None:
                self.right = cluster.left
            else:
                self.right = cluster
            self.values += cluster.values
            
        else:
            node = Cluster()
            node.left = self.left
            node.right = self.right
            node.values = self.values
            self.left = node
            if cluster.right is None:
                self.right = cluster.left
            else:
                self.right = cluster
            self.values += cluster.values

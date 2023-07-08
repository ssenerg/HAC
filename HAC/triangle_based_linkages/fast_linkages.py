from HAC.heaps.fibonacci import MaxFibonacciHeap
from typing import Tuple, List, Union
from HAC.tools.cluster import Cluster
from abc import ABC, abstractmethod
import numpy as np


class Linkage(ABC):

    """
    Linkage
    ------------------

    This class is Base Fast Linkage class

    Params:
        similarity (np.ndarray): The similarity matrix
        one_component (bool): True, if the similarity matrix has one component
    """

    def __init__(
            self, 
            similarity: np.ndarray, 
            one_component: bool = True
        ) -> "Linkage":

        # Error handling
        if not isinstance(similarity, np.ndarray):
            raise TypeError('Similarity matrix must be a numpy array.')
        if similarity.ndim != 2:
            raise ValueError('Similarity matrix must be two-dimensional.')
        if similarity.shape[0] != similarity.shape[1]:
            raise ValueError('Similarity matrix must be square.')
        if not isinstance(one_component, bool):
            raise TypeError('one_component must be a boolean object.')
        
        maximum, minimum = np.nanmax(similarity), np.nanmin(similarity)
        if np.isnan(maximum) or np.isnan(minimum):
            raise ValueError('Similarity matrix is full NaN.')
        
        if maximum > 1 or minimum < 0:
            raise ValueError('Indices of Similarity matrix must between 0 and 1.')

        self.complete = False if np.isnan(similarity).any() else True
        self.one_component = one_component
        if self.complete:
            self.one_component = True
        
        self.similarity = similarity
        self.clusters, self.heaps, self.hash_tables = self.__make_initials()

    def __make_initials(
            self
        ) -> Tuple[List[Cluster], List[MaxFibonacciHeap], List[dict]]:

        """
        This function is used to make initial clusters, heaps and hash tables
        
        ------------------
        
        Returns:
            clusters (List[Cluster]): The initial clusters
            heaps (List[MaxFibonacciHeap]): The initial heaps
            hash_tables (List[dict]): The initial hash tables
        """

        clusters, heaps, hash_tables = [], [], []

        if self.complete:
            for i in range(self.similarity.shape[0]):

                # Make initial clusters, heaps and hash tables
                clusters.append(Cluster(i))
                heaps.append(MaxFibonacciHeap())
                hash_tables.append(dict())

                for j in range(self.similarity.shape[1]):
                    if i == j:
                        continue
                    similarity = self.similarity[i, j]
                    heap_node = heaps[i].push(similarity, j)
                    hash_tables[i][j] = (similarity, heap_node)
        else:
            for i in range(self.similarity.shape[0]):

                # Make initial clusters, heaps and hash tables
                clusters.append(Cluster(i))
                heaps.append(MaxFibonacciHeap())
                hash_tables.append(dict())

                for j in range(self.similarity.shape[1]):
                    if i == j:
                        continue
                    similarity = self.similarity[i, j]
                    if not np.isnan(similarity):
                        heap_node = heaps[i].push(similarity, j)
                        hash_tables[i][j] = (similarity, heap_node)

        return clusters, heaps, hash_tables
        
    def chain_based(self) -> Union[Cluster, List[Cluster]]:

        """
        This function is used to compute the clusters
        using chain based algorithm

        ------------------

        Returns:
            Cluster: if the similarity matrix has one component
            List[Cluster]: if the similarity matrix has more than one component

        """

        if self.one_component:
            if self.complete:
                return self.__chain_based_complete_one_component()
            return self.__chain_based_uncomplete_one_component()
        return self.__chain_based_more_components()

    def __chain_based_complete_one_component(self) -> Cluster:

        """
        This function is used to compute the clusters
        using chain based algorithm when the similarity matrix
        has one component and is complete
        
        ------------------
        
        Returns:
            Cluster: The final cluster
        """

        cluster_id, length = 0, self.similarity.shape[0]

        while cluster_id < length:

            if not self.clusters[cluster_id].activate:
                cluster_id += 1
                continue

            stack = (cluster_id, )
            done = False

            while stack:

                top = stack[-1]
                if self.heaps[top].find_max is None:
                    done = True
                    break
                best_top = self.heaps[top].find_max.key
                if len(stack) > 1 and best_top == stack[-2]:
                    stack = stack[:-2]
                    self.__fast_merge_complete(top, best_top)
                else:
                    stack = stack + (best_top, )
            
            if done:
                return self.clusters[cluster_id]
    
    def __chain_based_uncomplete_one_component(self) -> Cluster:

        """
        This function is used to compute the clusters
        using chain based algorithm when the similarity matrix
        has one component and is uncomplete
        
        ------------------
        
        Returns:
            Cluster: The final cluster
        """
        
        cluster_id, length = 0, self.similarity.shape[0]

        while cluster_id < length:

            if not self.clusters[cluster_id].activate:
                cluster_id += 1
                continue

            stack = (cluster_id, )
            done = False

            while stack:

                top = stack[-1]
                if self.heaps[top].find_max is None:
                    done = True
                    break
                best_top = self.heaps[top].find_max.key
                if len(stack) > 1 and best_top == stack[-2]:
                    stack = stack[:-2]
                    self.__fast_merge_uncomplete(top, best_top)
                else:
                    stack = stack + (best_top, )
            
            if done:
                return self.clusters[cluster_id]

    def __chain_based_more_components(self) -> List[Cluster]:

        """
        This function is used to compute the clusters
        using chain based algorithm when the similarity matrix
        has more than one component
        
        ------------------
        
        Returns:
            List[Cluster]: The final clusters
        """
        
        cluster_id, length = 0, self.similarity.shape[0]
        final_clusters = []

        while cluster_id < length:

            if not self.clusters[cluster_id].activate:
                cluster_id += 1
                continue

            stack = (cluster_id, )

            while stack:

                top = stack[-1]
                if self.heaps[top].find_max is None:
                    final_clusters.append(self.clusters[cluster_id])
                    cluster_id += 1
                    break
                best_top = self.heaps[top].find_max.key
                if len(stack) > 1 and best_top == stack[-2]:
                    stack = stack[:-2]
                    self.__fast_merge_uncomplete(top, best_top)
                else:
                    stack = stack + (best_top, )
            
        return final_clusters

    def __fast_merge_complete(
            self, 
            key_a: int, 
            key_b: int
        ) -> None:

        """
        This function is used to merge two clusters
        when the similarity matrix is complete

        ------------------

        Params:
            key_a (int): The first cluster key
            key_b (int): The second cluster key
        """

        # delete the keys from hash tables and heaps
        del self.hash_tables[key_a][key_b]
        del self.hash_tables[key_b][key_a]
        self.heaps[key_a].pop()
        self.heaps[key_b].pop()

        intersect_tuple, hash_table_b = self.__t_merge_complete(key_a, key_b)

        for key_c, location_in_a, location_in_b in intersect_tuple:

            self.heaps[key_a].delete_node(location_in_a)
            self.heaps[key_b].delete_node(location_in_b)

            similarity = hash_table_b[key_c][0]
            hash_table_b[key_c] = (
                similarity, 
                self.heaps[key_b].push(similarity, key_c)
            )

        # merge the heaps and clusters
        self.clusters[key_b].merge(self.clusters[key_a])
        
        for key_c, _, _ in intersect_tuple:

            hash_table_c = self.hash_tables[key_c]
            heap_c = self.heaps[key_c]

            heap_c.delete_node(hash_table_c[key_a][1])
            del hash_table_c[key_a]

            new_similarity = hash_table_b[key_c][0]
            heap_c.delete_node(hash_table_c[key_b][1])
            hash_table_c[key_b] = (
                new_similarity, 
                heap_c.push(new_similarity, key_b)
            )

    def __fast_merge_uncomplete(
            self, 
            key_a: int, 
            key_b: int
        ) -> None:

        """
        This function is used to merge two clusters
        when the similarity matrix is uncomplete
        
        ------------------

        Params:
            key_a (int): The first cluster key
            key_b (int): The second cluster key
        """

        # change the keys if the first cluster has more nodes
        if self.heaps[key_a].total_nodes > self.heaps[key_b].total_nodes:
            key_a, key_b = key_b, key_a

        # delete the keys from hash tables and heaps
        del self.hash_tables[key_a][key_b]
        del self.hash_tables[key_b][key_a]
        self.heaps[key_a].pop()
        self.heaps[key_b].pop()
        
        intersect_tuple, minus_tuple, hash_table_b = self.__t_merge_uncomplete(key_a, key_b)

        for key_c, location_in_a, location_in_b in intersect_tuple:

            self.heaps[key_a].delete_node(location_in_a)
            self.heaps[key_b].delete_node(location_in_b)

            similarity = hash_table_b[key_c][0]
            hash_table_b[key_c] = (
                similarity, 
                self.heaps[key_b].push(similarity, key_c)
            )
            
        # merge the heaps and clusters
        self.heaps[key_b].merge(self.heaps[key_a])
        self.clusters[key_b].merge(self.clusters[key_a])

        for key_c in minus_tuple:

            hash_table_c = self.hash_tables[key_c]
            similarity, location = hash_table_c[key_a]
            location.key = key_b

            del hash_table_c[key_a]
            hash_table_c[key_b] = (similarity, location)
        
        for key_c, _, _ in intersect_tuple:

            hash_table_c = self.hash_tables[key_c]
            heap_c = self.heaps[key_c]

            heap_c.delete_node(hash_table_c[key_a][1])
            del hash_table_c[key_a]

            new_similarity = hash_table_b[key_c][0]
            heap_c.delete_node(hash_table_c[key_b][1])
            hash_table_c[key_b] = (
                new_similarity, 
                heap_c.push(new_similarity, key_b)
            )

    def __t_merge_complete(
            self, 
            key_a: int, 
            key_b: int
        ) -> None:

        """
        This function is used to merge two clusters
        when the similarity matrix is complete
        
        ------------------
        
        Params:
            key_a (int): The first cluster key
            key_b (int): The second cluster key
        """

        hash_table_a = self.hash_tables[key_a]
        hash_table_b = self.hash_tables[key_b]

        intersect_tuple = tuple()
        
        for key, value in hash_table_a.items():

            similarity = self._linkage_measure(key_a, key_b, key)
            intersect_tuple = \
                intersect_tuple + ((key, value[1], hash_table_b[key][1]), )
            hash_table_b[key] = (similarity, None)
        
        return intersect_tuple, hash_table_b

    def __t_merge_uncomplete(
            self, 
            key_a: int, 
            key_b: int
        ) -> None:

        """
        This function is used to merge two clusters
        when the similarity matrix is uncomplete
        
        ------------------
        
        Params:
            key_a (int): The first cluster key
            key_b (int): The second cluster key
        """
        
        hash_table_a = self.hash_tables[key_a]
        hash_table_b = self.hash_tables[key_b]

        intersect_tuple, minus_tuple = tuple(), tuple()

        for key, value in hash_table_a.items():

            if key in hash_table_b:
                similarity = self._linkage_measure(key_a, key_b, key)
                intersect_tuple = \
                    intersect_tuple + ((key, value[1], hash_table_b[key][1]), )
                hash_table_b[key] = (similarity, None)

            else:
                hash_table_b[key] = value
                minus_tuple = minus_tuple + (key, )

        return intersect_tuple, minus_tuple, hash_table_b

    @abstractmethod
    def _linkage_measure(
        self, 
        key_a: int, 
        key_b: int, 
        key_c: int
    ) -> float:
        pass


class SingleLinkage(Linkage):

    """
    Single Linkage
    ------------------
    
    This class is Fast Single Linkage class
    which it linkage measure returns minimum 
    similarity between two clusters
    
    Params:
        similarity (np.ndarray): The similarity matrix
        one_component (bool): True, if the similarity matrix has one component
    """

    def _linkage_measure(
            self, 
            key_a: int, 
            key_b: int, 
            key_c: int
        ) -> float:

        """
        This function is used to compute the linkage measure
        between two clusters
        
        ------------------
        
        Params:
            key_a (int): The first cluster key
            key_b (int): The second cluster key
            key_c (int): The third cluster key which is going 
            to be merged with the first and second clusters
        Returns:
            float: The linkage measure value
        """
        
        values_a_b = self.clusters[key_a].values + self.clusters[key_b].values
        values_c = self.clusters[key_c].values

        return np.nanmin(self.similarity[values_a_b][:, values_c])


class CompleteLinkage(Linkage):

    """
    Complete Linkage
    ------------------
    
    This class is Fast Complete Linkage class
    which it linkage measure returns maximum
    similarity between two clusters
    
    Params:
        similarity (np.ndarray): The similarity matrix
        one_component (bool): True, if the similarity matrix has one component
    """

    def _linkage_measure(
            self, 
            key_a: int, 
            key_b: int, 
            key_c: int
        ) -> float:
        
        """
        This function is used to compute the linkage measure
        between two clusters
        
        ------------------
        
        Params:
            key_a (int): The first cluster key
            key_b (int): The second cluster key
            key_c (int): The third cluster key which is going 
            to be merged with the first and second clusters
        Returns:
            float: The linkage measure value
        """
        
        values_a_b = self.clusters[key_a].values + self.clusters[key_b].values
        values_c = self.clusters[key_c].values
        
        return np.nanmax(self.similarity[values_a_b][:, values_c])

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import typing
import networkx as nx


class Network:

    '''
    networkx graph of connected neurons from
    layer to layer
    '''

    def __init__(self, layers: list = None):
        '''
        input: a list of layer objects
        creates: empty multi-graph
        '''
        self.layers = layers
        self.graph = nx.MultiGraph()
        self.edge_dict = defaultdict()

    def int_graph(self, layers: list = None):
        '''
        create networkx graph between layers
        '''

        if layers:
            self.layers = layers

        nodes = 0
        # traverse layers
        for layer in self.layers:
            # traverse nodes in each layer
            for n in layer.neurons:
                n.set_id(nodes)
                self.graph.add_node(n)
                nodes += 1

    def layers_cycle(self, k: int = 5):
        '''
        update connections of neurons
        from layer to layer
        '''

        for l_idx, layer in enumerate(self.layers[:-1]):
            for idx, n in enumerate(layer.neurons):
                self.edge_dict[idx] = n.edges_delta(k, self.layers[l_idx + 1])
                sl = sorted(self.edge_dict[idx])
                for e in sl[0:k]:
                    self.graph.add_edge(
                        idx, e[1].id, weight=abs(n.input - e[1].input))

import numpy as np
import numpy.random as rdm
from tqdm import tqdm
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt

class UGraph:
    # Graphs are generated in adjacency lists. Each index of the list correpond to the same index of the node of the graph. And the contents of the list
    # are the nodes to which the node is connected.
    # Suppose is a undirected graph.

    def __init__(self, n, p):
        if n < 2:
            raise Exception('Nodes need to be greater than 2')
            n = 2
        self.nodes_len = n
        self.p = p
        self.nodes = []
        self.G, self.G_deg = self.gen_random_graph(n,p)
        self.index = 0
        self.convex = False
        if n == self.BFS(self.G):
            self.convex = True

    def gen_random_graph(self, n, p):
        G = []
        cards = [] # Cardinality of each nodes or degree
        self.nodes = [i for i in range(n)]

        for node in self.nodes:
            new_edges = []
            card = 0
            for i in self.nodes:
                if i == node:
                    None # No autoconnections allowed
                elif p >= rdm.random_sample():
                    new_edges.append(i)
                    card += 1
            G.append(new_edges)
            cards.append(card)

        return (G, cards)

    def __iter__(self):
        return self
    def __next__(self):
        if self.index == self.nodes_len:
            raise StopIteration
        i = self.index
        self.index += 1
        return (i, self.G[i], self.G_deg[i])
    # Returning the actual node, the nodes where it goes to and the degree of those
    def __repr__(self):
        return 'Undirected graph with {0} nodes. Convexity {1}'.format(self.nodes_len,self.convex)

    @staticmethod
    def BFS(T, s=None, ls=None):
        # T is the graph to discover. It can be feed as a adjacency list for indexes from 0 to nodes - 1
        # S is the starting node, is None, then takes the first item of T
        # ls is the number of nodes
        if ls == None:
            ls = len(T)
        if s == None:
            s = 0
        discovered = np.zeros(ls, dtype=np.bool_)
        discovered[s] = True
        BFS_len = 1
        L = [s]
        while L != []:
            L_1 = []
            for u in L:
                for v in T[u]:
                    if not discovered[v]:
                        discovered[v] = True
                        L_1.append(v)
                        BFS_len += 1
            L = L_1
        return BFS_len

def evaluate_graph(n, pr, i, workers):
    with Pool(processes=workers) as p:
        positives = p.map(evaluate_graph_sub, [(n,pr) for _ in range(i)])
        p.close()
        p.join()
    positive = np.sum(positives)
    print(pr, positive)
    return positive/i

def evaluate_graph_sub(SS):
    n, p = SS
    graph = UGraph(n,p)
    if graph.convex:
        return 1
    else:
        return 0

import os

tests = []
iterations = 10
workers=os.cpu_count()
n_test = np.linspace(10**3, 8*10**3, num=6, dtype=np.int32)
p_test = np.linspace(0.001, 0.002, num=30)
name_file = 'random_graphs_results.npy'


if __name__ == '__main__':
    files = os.listdir()
    if name_file in files:
        try:
            #n_test, p_test, tests = np.load(name_file)
            n_test, p_test, tests = np.load(name_file, allow_pickle=True)
        except:
            print("No success loading the file. Exists but failed.")
    else:
        for n in tqdm(n_test):
            test = []
            for p in p_test:
                test.append(evaluate_graph(n,p,iterations,workers))
            tests.append(test)
        np.save(name_file, (n_test, p_test, np.array(tests)))
    #print(tests)
    fig, axs = plt.subplots(figsize=(0.5*len(p_test)+1,0.7*len(n_test)+2))
    heat = sns.heatmap(np.array(tests), vmin=0, ax=axs, xticklabels=p_test, yticklabels=n_test)
    axs.set(xlabel='$p$', ylabel='$n$', title="Average on %d iterations for convexity on random graphs."%(  iterations))
    plt.show()

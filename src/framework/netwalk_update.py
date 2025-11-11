import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import pandas as pd
import os
import glob

class Reservior:
    """
    有向 + 有权 + 支持“全量月度更新”的动态Reservior版本
    维护每个节点的出边样本（reservoir）
    """
    def __init__(self, edges, vertices, dim=10, seed=24):
        self.reservior = {}      # 每个节点的邻居样本
        self.degree = {}
        self.weight_sum = {}
        self.edge_weights = {}   # 存储实际边的权重
        self.init_edges = edges
        self.vertices = vertices
        self.reservior_dim = dim
        self.seed = seed
        self.__build()

    def __build(self):
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.init_edges)
        for v in self.vertices:
            if v in g:
                nbrs = list(g.successors(v))
                weights = np.array([g[v][n]['weight'] for n in nbrs])
                np.random.seed(self.seed)
                if len(nbrs) > 0:
                    probs = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                    indices = np.random.choice(len(nbrs), size=self.reservior_dim, p=probs)
                    self.reservior[v] = np.array([nbrs[idx] for idx in indices])
                    self.edge_weights[v] = {n: g[v][n]['weight'] for n in nbrs}
                    self.degree[v] = len(nbrs)
                    self.weight_sum[v] = weights.sum()
                else:
                    self.reservior[v] = np.array([None]*self.reservior_dim)
                    self.edge_weights[v] = {}
                    self.degree[v] = 0
                    self.weight_sum[v] = 0
            else:
                self.reservior[v] = np.array([None]*self.reservior_dim)
                self.edge_weights[v] = {}
                self.degree[v] = 0
                self.weight_sum[v] = 0

    def update(self, edges_current):
        if edges_current is None or len(edges_current) == 0:
            return
        self.remove_missing_edges(edges_current)
        rng = np.random.default_rng(self.seed)
        for u, v, w in tqdm(edges_current, desc="Updating edges"):
            if u not in self.edge_weights:
                self.edge_weights[u] = {}
            if u not in self.weight_sum:
                self.weight_sum[u] = 0
            if u not in self.degree:
                self.degree[u] = 0
            if u not in self.reservior:
                self.reservior[u] = np.array([None]*self.reservior_dim)

            old_weight = self.edge_weights[u].get(v, 0)
            self.edge_weights[u][v] = w
            self.weight_sum[u] += (w - old_weight)
            self.degree[u] = len(self.edge_weights[u])

            if self.weight_sum[u] > 0:
                nbrs = list(self.edge_weights[u].keys())
                weights = np.array(list(self.edge_weights[u].values()), dtype=float)
                probs = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
                indices = rng.choice(len(nbrs), size=self.reservior_dim, p=probs)
                self.reservior[u] = np.array([nbrs[idx] for idx in indices])

    def remove_missing_edges(self, edges_current):
        current_set = {(u,v) for (u,v,_) in edges_current}
        for u in list(self.edge_weights.keys()):
            for v in list(self.edge_weights[u].keys()):
                if (u,v) not in current_set:
                    old_w = self.edge_weights[u][v]
                    del self.edge_weights[u][v]
                    self.weight_sum[u] -= old_w
                    self.degree[u] -= 1

        for u in self.reservior.keys():
            valid_neighbors = [n for n in self.reservior[u] if n in self.edge_weights.get(u, {})]
            if len(valid_neighbors) == 0:
                self.reservior[u] = np.array([None]*self.reservior_dim)
            else:
                self.reservior[u] = np.array(valid_neighbors[:self.reservior_dim])

class WalkUpdate:
    def __init__(self, init_edges, vertices, walk_len=3, walk_per_node=5, seed=24):
        self.init_edges = init_edges
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        self.seed = seed
        self.reservior = Reservior(edges=self.init_edges, vertices=vertices)
        self.prev_walks = self.__init_walks()
        self.new_walks = None
        self.training_walks = None

    def __init_walks(self):
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.init_edges)
        rand = random.Random(self.seed)
        walks = []
        nodes = list(g.nodes())
        for _ in range(self.walk_per_node):
            rand.shuffle(nodes)
            for node in nodes:
                walks.append(self.__random_walk(g, node, rand=rand))
        return walks

    def __random_walk(self, g, start, alpha=0, rand=random.Random(0)):
        walk = [start]
        while len(walk) < self.walk_len:
            cur = walk[-1]
            nbrs = list(g.successors(cur))
            if len(nbrs) == 0:
                break
            weights = np.array([g[cur][nbr]['weight'] for nbr in nbrs], dtype=float)
            probs = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            if rand.random() >= alpha:
                next_node = rand.choices(nbrs, weights=probs, k=1)[0]
            else:
                next_node = start
            walk.append(next_node)
        while len(walk) < self.walk_len:
            walk.append(start)
        return walk

    def __generate(self, edges_current):
        walk_set = []
        rand = random.Random(self.seed)
        start_nodes = set([u for u, _, _ in edges_current])
        for n in start_nodes:
            if n not in self.reservior.reservior:
                continue
            for _ in range(self.walk_per_node):
                x = rand.choice(self.reservior.reservior[n])
                if x is None:
                    continue
                y_choices = self.reservior.reservior.get(x, [])
                y_choices = [y for y in y_choices if y is not None]
                if len(y_choices) == 0:
                    continue
                y = rand.choice(y_choices)
                walk_set.append([n, x, y])

        old_samples = [w for w in self.prev_walks if w[0] not in start_nodes]
        self.new_walks = walk_set
        self.training_walks = self.new_walks + old_samples
        self.prev_walks = self.training_walks
        print(f"length of training walks: {len(self.training_walks)}")
        return self.training_walks

    def update(self, edges_current):
        if edges_current is None or len(edges_current) == 0:
            return []  # 返回空列表，避免报错
        self.reservior.update(edges_current)
        return self.__generate(edges_current)

class NetWalk_update:
    def __init__(self, path, walk_per_node=5, walk_len=3, init_percent=0.5, snap=10, seed=24):
        self.seed = seed
        self.data_path = path
        self.walk_len = walk_len
        self.init_percent = init_percent
        self.snap = snap
        self.vertices = None
        self.idx = 0
        self.walk_per_node = walk_per_node

        # 读取数据
        if os.path.isdir(path):
            self.data = self.__get_data_from_folder()
        else:
            self.data = self.__get_data_from_file()

        init_edges, snapshots = self.data
        self.vertices = np.unique(np.vstack([init_edges]+snapshots)[:, :2]) if snapshots else np.unique(init_edges[:, :2])
        self.node2idx = {node: i for i, node in enumerate(self.vertices)}
        self.idx2node = {i: node for i, node in enumerate(self.vertices)}
        self.walk_update = WalkUpdate(init_edges, self.vertices, walk_len=self.walk_len, walk_per_node=self.walk_per_node, seed=self.seed)

    def __get_data_from_folder(self):
        files = sorted(glob.glob(os.path.join(self.data_path, "*.csv")))
        if not files:
            raise ValueError(f"No CSV files found in {self.data_path}")
        init_edges = pd.read_csv(files[0]).values
        snapshots = [pd.read_csv(f).values for f in files[1:]]
        print(f"total vertices: {len(np.unique(np.vstack([init_edges]+snapshots)[:, :2]))}, initial edges: {len(init_edges)}, snapshots: {len(snapshots)}")
        return init_edges, snapshots

    def __get_data_from_file(self):
        edges = np.loadtxt(self.data_path, dtype=str)
        init_idx = int(len(edges)*0.1)
        init_edges = edges[:init_idx]
        snapshots = []
        current = init_idx
        while current < len(edges):
            snapshots.append(edges[current:current+self.snap])
            current += self.snap
        print(f"total vertices: {len(np.unique(edges[:, :2]))}, initial edges: {len(init_edges)}, snapshots: {len(snapshots)}")
        return init_edges, snapshots

    def getOnehot(self, walks):
        walk_idx = [[self.node2idx[w] for w in walk] for walk in walks]
        walk_mat = np.array(walk_idx, dtype=int)
        rows = walk_mat.flatten()
        cols = np.arange(len(rows))
        data = np.ones(len(rows))
        coo = coo_matrix((data, (rows, cols)), shape=(len(self.vertices), len(rows)))
        onehot_walks = csr_matrix(coo)
        return onehot_walks.toarray()

    def hasNext(self):
        _, snapshots = self.data
        return self.idx < len(snapshots)

    def nextOnehotWalks(self):
        if not self.hasNext():
            return np.zeros((len(self.vertices), 0), dtype=int)
        _, snapshots = self.data
        snapshot = snapshots[self.idx]
        self.idx += 1
        walks = self.walk_update.update(snapshot)
        return self.getOnehot(walks)

    def getInitWalk(self):
        return self.getOnehot(self.walk_update.prev_walks)

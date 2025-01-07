from multiprocessing import Pool
import time
import itertools

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, weight, processes=12):
    """Parallel betweenness centrality  function"""
    print("parallel calculation for betweenness...")
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G.nodes(), G.order() // node_divisor))
    num_chunks = len(node_chunks)
    items = list(zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [False] * num_chunks,
            [weight] * num_chunks,
            ))

    bt_sc = p.starmap(
        nx.edge_betweenness_centrality_subset,
        tqdm(
            items,
        ),
    )
    p.close()
    p.join()

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in tqdm(bt_sc[1:]):
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c
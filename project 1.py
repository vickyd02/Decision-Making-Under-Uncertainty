import sys
import os
import random

import networkx as nx
import time
import pandas as pd
import numpy as np
from scipy.special import gammaln


def load_data(path):
    """Read a CSV and return variable names, data array, number of values per variable"""
    df = pd.read_csv(path)
    names = list(df.columns)
    data = df.values - 1
    r = data.max(axis=0) + 1
    return names, data, r


# The Bayesian Score, using uniform Dirichlet prior with an alpha of 1

def family_score(i, parents, data, r):
    """Score contribution of variable i given a list of its parent indices."""
    ri = r[i]
    if parents:
        # one integer per row encoding its parent assignment j
        # every row is a single number that identifies which combo
        # of parent values appear in the rows
        j = np.ravel_multi_index(data[:, parents].T, r[parents])
    else:
        j = np.zeros(len(data), dtype=np.int64)
    jk = j * ri + data[:, i]                     # create the (j, k) pairs
    _, m_jk = np.unique(jk, return_counts=True)  # counts m_ijk
    _, m_j0 = np.unique(j, return_counts=True)   # counts m_ij0
    # bayesian score formula with every pseudocount alpha = 1
    # combinations that do not appear in the data are not counted at all
    return (gammaln(ri) - gammaln(ri + m_j0)).sum() + gammaln(1 + m_jk).sum()


def bayesian_score(parents, data, r):
    # totals the score for every variable and gives the total score
    return sum(family_score(i, parents[i], data, r) for i in range(len(r)))


# Actual K2 search

def k2(data, r, ordering, max_parents=3):
    """K2: for each variable, add the best parent from the variables
    that come prior in the ordering so the graph does not cycle."""
    parents = {i: [] for i in range(len(r))}
    for pos, i in enumerate(ordering):
        best = family_score(i, parents[i], data, r)
        while len(parents[i]) < max_parents:
            best_candidate, best_new = None, best
            for p in ordering[:pos]:              # only prior vars
                if p in parents[i]:               # skip if parent is already in the list
                    continue
                s = family_score(i, parents[i] + [p], data, r)
                if s > best_new:                  # if the new score is better, update the best candidate
                    best_candidate, best_new = p, s
            if best_candidate is None:            # no parent improves the score
                break
            parents[i].append(best_candidate)     # add the best candidate to the list
            best = best_new                       # update the best score
    return parents


# Stage 1: K2 with many random orderings

def k2_random_restarts(data, r, n_restarts=50, max_parents=3, seed=0):
    """Run K2 with random orderings and keep the best graph."""
    rng = random.Random(seed)
    n = len(r)
    best_graph, best_score = None, float("-inf")
    for _ in range(n_restarts):
        ordering = list(range(n))
        rng.shuffle(ordering)
        graph = k2(data, r, ordering, max_parents)
        score = bayesian_score(graph, data, r)
        if score > best_score:
            best_graph, best_score = graph, score
    return best_graph, best_score


# Stage 2: local search (hill climbing)

def creates_cycle(parents, a, b):
    """Would adding edge a -> b create a cycle?
    Yes if b is already an ancestor of a (b -> ... -> a)."""
    stack, seen = [a], set()
    while stack:
        node = stack.pop()
        if node == b:
            return True
        if node not in seen:
            seen.add(node)
            stack.extend(parents[node])
    return False


def hill_climb(parents, data, r, max_parents=3):
    """Repeatedly make the single best edge change (add, remove, or reverse)
    until no change improves the score."""
    parents = {i: list(ps) for i, ps in parents.items()}   # copy
    n = len(r)
    cache = {}

    def fs(i, ps):  # cached family score (score doesn't depend on parent order)
        key = (i, tuple(sorted(ps)))
        if key not in cache:
            cache[key] = family_score(i, list(key[1]), data, r)
        return cache[key]

    while True:
        best_gain, best_move = 0, None
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if a in parents[b]:
                    # Try removing a -> b
                    new_b = [p for p in parents[b] if p != a]
                    gain = fs(b, new_b) - fs(b, parents[b])
                    if gain > best_gain:
                        best_gain, best_move = gain, ("remove", a, b)
                    # Try reversing a -> b into b -> a
                    if len(parents[a]) < max_parents:
                        parents[b].remove(a)                 # temporarily remove
                        ok = not creates_cycle(parents, b, a)
                        parents[b].append(a)                 # put it back
                        if ok:
                            gain = (fs(b, new_b) - fs(b, parents[b])
                                    + fs(a, parents[a] + [b]) - fs(a, parents[a]))
                            if gain > best_gain:
                                best_gain, best_move = gain, ("reverse", a, b)
                elif len(parents[b]) < max_parents and not creates_cycle(parents, a, b):
                    # Try adding a -> b
                    gain = fs(b, parents[b] + [a]) - fs(b, parents[b])
                    if gain > best_gain:
                        best_gain, best_move = gain, ("add", a, b)

        if best_move is None:          # no improving move left: done
            return parents
        move, a, b = best_move
        if move == "add":
            parents[b].append(a)
        elif move == "remove":
            parents[b].remove(a)
        else:  # reverse
            parents[b].remove(a)
            parents[a].append(b)


# Actual graphing

def write_gph(parents, idx2names, filename):
    # for every arrow this writes one line for parent and child
    with open(filename, 'w') as f:
        for child, ps in parents.items():
            for p in ps:
                f.write("{},{}\n".format(idx2names[p], idx2names[child]))


def run(name, n_restarts=50, max_parents=3):
    folder = os.path.dirname(os.path.abspath(__file__))
    names, data, r = load_data(os.path.join(folder, f"{name}.csv"))
    start = time.time()
    graph, k2_score = k2_random_restarts(data, r, n_restarts, max_parents)
    graph = hill_climb(graph, data, r, max_parents)
    elapsed = time.time() - start          # search time -> to use in pdf
    score = bayesian_score(graph, data, r)
    write_gph(graph, {i: n for i, n in enumerate(names)},
              os.path.join(folder, f"{name}.gph"))
    print(f"{name}: K2 best = {k2_score:.2f}, after hill climbing = {score:.2f}, "
          f"time = {elapsed:.2f} s")
    return score


if __name__ == "__main__":
    # (n_restarts, max_parents) tuned per dataset
    settings = {"small": (100, 6), "medium": (200, 8), "large": (100, 8)}
    total = 0
    for name, (n_restarts, max_parents) in settings.items():
        total += run(name, n_restarts, max_parents)
    print(f"combined score = {total:.2f}")
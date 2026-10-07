# CS238: Decision Making Under Uncertainty

**Stanford University · Autumn 2026**

Coursework for Stanford's graduate course on making optimal decisions when outcomes are uncertain, information is incomplete, and the environment has to be learned from data. The course covers probabilistic modeling, Markov decision processes, reinforcement learning, and planning under partial observability, following *Algorithms for Decision Making* (Kochenderfer, Wheeler & Wray).

| Project | Topic | Core Methods | Status |
|---|---|---|---|
| [1](#project-1-bayesian-network-structure-learning) | Bayesian Network Structure Learning | K2 search, hill climbing, Bayesian scoring | ✅ Complete |
| [2](#project-2-reinforcement-learning) | Reinforcement Learning | TODO | 🚧 In progress |
| [3](#project-3-final-project) | Final Project | TODO | 🚧 Planned |

**Tech stack:** Python · NumPy · SciPy · pandas · NetworkX · Graphviz

---

## Project 1: Bayesian Network Structure Learning

Given raw observational data and no prior knowledge of how the variables relate, this project learns the dependency structure between them: which variables directly influence which. The output is a Bayesian network, a directed acyclic graph that serves as an interpretable probabilistic model of the system.

### Results

| Dataset | Variables | Samples | Baseline (no edges) | My Model | Improvement | Search Time |
|---|---|---|---|---|---|---|
| Titanic | 8 | 889 | −4,166 | **−3,799** | 8.8% | 0.3 s |
| Wine Quality | 13 | 6,497 | −117,669 | **−96,542** | 18.0% | 6.6 s |
| Black-box system | 50 | 10,000 | −524,395 | **−457,700** | 12.7% | 69 s |

The 50-variable dataset was anonymized: no variable names, no domain context, just data. On a problem that size, exhaustive search is impossible, because the number of candidate graphs exceeds the number of atoms in the universe.

### Approach

**1. Bayesian scoring from scratch.** I implemented the Bayesian score under a uniform Dirichlet prior without any structure-learning libraries. The scorer encodes each parent configuration as a single integer index, so every family score comes down to two vectorized counting operations. Because the score decomposes by variable, each candidate change only requires rescoring the one or two variables it affects.

**2. K2 search with random restarts.** K2 greedily assigns each variable the parents that most improve its score, restricted to variables earlier in a fixed ordering. This guarantees an acyclic graph by construction, but the result depends heavily on the ordering. Running K2 on many random orderings and keeping the best result removes much of that dependence.

**3. Hill-climbing refinement.** Starting from the best K2 graph, a local search evaluates every possible single-edge addition, removal, and reversal, and applies the best one until no move helps. This frees the model from K2's ordering constraint. A depth-first search rejects any move that would create a cycle, and memoized family scores keep each pass fast.

**4. Hyperparameter tuning.** Sweeping the maximum number of parents per variable from 3 to 15 showed that the 50-variable system contains genuinely high-order dependencies. Raising the cap from 3 to 8 improved the score by more than 15,000 points, and the gain plateaued beyond 8.

### Validation

- The scorer reproduces the course's reference score exactly (−132.57689).
- It matches an independent line-by-line translation of the textbook's Julia implementation on all three datasets.
- Every output graph is verified to be acyclic, and every node is checked against its parent limit.

### Why It Matters

Learning structure from data is central to working with complex physical systems where the full set of interactions isn't known in advance. Examples include sensor networks, process lines, and power grids. This approach:

- **Discovers which variables actually drive which**, rather than fitting a black-box predictor.
- **Produces interpretable models** that engineers can inspect, question, and reason about.
- **Scales sensibly**, since decomposable scoring and caching make a 50-variable search run in about a minute on a laptop.
- **Supports downstream decisions under uncertainty**, because a learned Bayesian network can answer "what happens to Y if X changes?" with calibrated probabilities.

### Running It

```bash
cd project1
pip install pandas numpy scipy networkx
python project_1.py        # learns all three graphs, writes small.gph, medium.gph, large.gph
python plot_graphs.py      # renders graph visualizations (requires Graphviz)
```

---

## Project 2: Reinforcement Learning

<!-- TODO: One or two sentences on the problem. What is the agent trying to learn, and from what data? -->

### Results

<!-- TODO: Fill in once complete. -->

| Dataset | States | Actions | Baseline | My Policy | Improvement | Run Time |
|---|---|---|---|---|---|---|
| Small | TODO | TODO | TODO | **TODO** | TODO | TODO |
| Medium | TODO | TODO | TODO | **TODO** | TODO | TODO |
| Large | TODO | TODO | TODO | **TODO** | TODO | TODO |

### Approach

<!-- TODO: Describe each stage of your method, one bolded step per paragraph, like Project 1. -->

**1. TODO.**

**2. TODO.**

**3. TODO.**

### Validation

<!-- TODO: How did you check your implementation was correct? -->

- TODO

### Why It Matters

<!-- TODO: Connect the method to real-world decision problems (e.g., control, scheduling, resource allocation). -->

### Running It

```bash
cd project2
# TODO: install and run commands
```

---

## Project 3: Final Project

<!-- TODO: Project title, team members (if any), and a short summary of the problem. -->

### Problem

<!-- TODO: What decision problem did you tackle, and why is it hard? -->

### Approach

<!-- TODO: Model formulation (states, actions, observations, rewards) and solution method. -->

### Results

<!-- TODO: Key results, figures, and comparisons against baselines. -->

### Why It Matters

<!-- TODO: Real-world relevance. -->

### Running It

```bash
cd project3
# TODO: install and run commands
```

---

## Repository Structure

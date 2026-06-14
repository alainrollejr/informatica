"""
create_dom_notebook.py
Generates dominating_sets.ipynb using json.dump.
Run: python create_dom_notebook.py
"""
import json

def code(src, cid):
    return {"cell_type": "code", "execution_count": None,
            "id": cid, "metadata": {}, "outputs": [], "source": src}

def md(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

# ================================================================== cell sources

TITLE = """\
# Graph — Undirected Graphs and Minimum Dominating Sets

This notebook focuses on the `Graph` base class and its most
computationally interesting method: `minimum_dominating_sets()`.

| Topic | What we cover |
|---|---|
| **Graph class** | internal `_adj` structure, basic operations |
| **Dominating Set** | definition, intuition, real-world motivation |
| **Algorithm** | backtracking with two pruning rules |
| **Examples** | path, star, sensor network, complex graph |
| **Debug mode** | step-by-step algorithm trace |

---"""

IMPORTS = """\
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from graphs import Graph, format_solutions

%matplotlib inline
plt.rcParams['figure.dpi'] = 110"""

DRAW_FN = """\
# ── Colour palette ────────────────────────────────────────────────
CLR_NODE = '#3A86D4'    # blue  — default node
CLR_HIGH = '#D94F4F'    # red   — highlighted node / relay station
CLR_EDGE = '#BBBBBB'    # grey  — default edge
CLR_EEDG = '#D94F4F'    # red   — coverage edge


def _positions(graph):
    \"\"\"
    Compute (x, y) positions for each node.
    Each connected component gets its own circle; components are
    spaced horizontally so they never overlap.
    \"\"\"
    components = graph.connected_components()
    n_comps    = len(components)
    pos        = {}
    cx_offset  = 0.0

    for comp in components:
        n = len(comp)
        r = max(1.0, n * 0.45)          # radius scales with component size
        for i, node in enumerate(sorted(comp)):
            if n == 1:
                pos[node] = (cx_offset, 0.0)
            else:
                angle = 2 * math.pi * i / n - math.pi / 2
                pos[node] = (cx_offset + r * math.cos(angle),
                             r * math.sin(angle))
        cx_offset += 2 * r + 1.5        # gap between components

    # Re-centre around x=0
    xs = [p[0] for p in pos.values()]
    mid = (min(xs) + max(xs)) / 2
    pos = {n: (x - mid, y) for n, (x, y) in pos.items()}
    return pos


def draw_graph(graph, title='Graph',
               highlight_nodes=None,
               highlight_edges=None,
               legend=None):
    \"\"\"
    Draw an undirected Graph.

    Layout: each connected component arranged in a circle.
    Disconnected components are placed side by side.

    highlight_nodes : list of node names to draw in red
    highlight_edges : list of (a, b) tuples to draw in red
    legend          : list of (colour, label) tuples
    \"\"\"
    hi_nodes = set(highlight_nodes or [])
    hi_edges = {tuple(sorted(e)) for e in (highlight_edges or [])}

    nodes = graph.nodes()
    if not nodes:
        print('(empty graph)')
        return

    pos   = _positions(graph)
    xs    = [p[0] for p in pos.values()]
    ys    = [p[1] for p in pos.values()]
    pad   = 1.0
    NODE_R = 0.50

    fig_w = max(6, (max(xs) - min(xs) + 2 * pad) * 1.3)
    fig_h = max(5, (max(ys) - min(ys) + 2 * pad) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')
    ax.set_aspect('equal')

    # ── Edges ─────────────────────────────────────────────────────
    drawn = set()
    for a in nodes:
        for b in graph.neighbors(a):
            key = tuple(sorted([a, b]))
            if key in drawn:
                continue
            drawn.add(key)
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            dx, dy = x1 - x0, y1 - y0
            length = (dx**2 + dy**2) ** 0.5
            if length < 1e-9:
                continue
            ux, uy = dx / length, dy / length
            is_hi  = key in hi_edges
            ax.plot([x0 + ux * NODE_R, x1 - ux * NODE_R],
                    [y0 + uy * NODE_R, y1 - uy * NODE_R],
                    color=CLR_EEDG if is_hi else CLR_EDGE,
                    lw=2.8 if is_hi else 1.5,
                    zorder=1)

    # ── Nodes ─────────────────────────────────────────────────────
    for n in nodes:
        x, y = pos[n]
        fc   = CLR_HIGH if n in hi_nodes else CLR_NODE
        ax.add_patch(plt.Circle((x, y), NODE_R, color=fc,
                                ec='white', lw=2, zorder=3))
        ax.text(x, y, n, ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

    # ── Legend ────────────────────────────────────────────────────
    if legend:
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in legend],
                  loc='lower right', fontsize=9, framealpha=0.85)

    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    plt.tight_layout()
    plt.show()


def draw_mds(graph, solution, title='Minimum Dominating Set', solution_index=None):
    \"\"\"
    Draw a Graph with one dominating set solution highlighted.

    Relay stations (in S) are red.
    Edges between a relay and a covered node are also red.
    The number of relay stations and which ones are labelled in the title.
    \"\"\"
    sol_set = set(solution)
    cov_edges = []
    covered_by = {}          # node -> relay that covers it (for annotation)

    for node in sorted(sol_set):
        for nbr in graph.neighbors(node):
            if nbr not in sol_set:
                key = tuple(sorted([node, nbr]))
                if key not in cov_edges:
                    cov_edges.append(key)
                covered_by[nbr] = node

    label = f'{sorted(solution)}'
    if solution_index is not None:
        label = f'Solution {solution_index}: {label}'
    full_title = f'{title}\\n{label}  (size = {len(solution)})'

    draw_graph(graph,
               title=full_title,
               highlight_nodes=list(sol_set),
               highlight_edges=cov_edges,
               legend=[(CLR_NODE, 'covered node (1 hop from relay)'),
                       (CLR_HIGH, 'relay station')])


print('draw_graph and draw_mds ready.')"""

MD_GRAPH = """\
---
## The `Graph` Class — Internal Structure

A `Graph` stores one central data structure:

```
_adj : dict[str, list[str]]
```

**`_adj[x]`** contains the list of all nodes directly connected to `x`.  
Because the graph is **undirected**, every edge A–B is stored *both ways*:
`_adj['A']` contains `'B'`  **AND**  `_adj['B']` contains `'A'`.

```
Graph(edges=[('A','B'), ('B','C'), ('A','C')])

_adj = {
    'A': ['B', 'C'],   ← A is connected to B and C
    'B': ['A', 'C'],   ← B is connected to A and C
    'C': ['B', 'A'],   ← C is connected to B and A
}
```

This symmetric storage makes `neighbors(node)` an O(1) lookup."""

CODE_GRAPH = """\
# Build a simple triangle and inspect the internal _adj dict
tri = Graph(edges=[('A','B'), ('B','C'), ('A','C')])

print('nodes()    :', tri.nodes())
print('edges()    :', tri.edges())
print('neighbors A:', tri.neighbors('A'))
print('degree A   :', tri.degree('A'))
print()
print('Internal _adj (the only data structure stored):')
for node in tri.nodes():
    print(f'  _adj[{node!r}] = {sorted(tri._adj[node])}')

draw_graph(tri, title='A simple triangle — Graph(edges=[(A,B),(B,C),(A,C)])')"""

MD_DOM = """\
---
## What is a Dominating Set?

A **dominating set** is a subset S of nodes such that:

> **Every node NOT in S has at least one direct neighbour IN S.**

In other words: from S you can *reach every other node in exactly one hop*.

**Real-world analogy:** S = relay stations in a wireless sensor network.  
Every sensor that is NOT a relay must be within one hop of a relay.

The **minimum dominating set** is the *smallest* such S.  
Finding it is **NP-complete** in general — no efficient algorithm is known  
that works on all graphs. We use **backtracking with two pruning rules** to  
find ALL optimal solutions exactly.

### Key concept: coverage

```
coverage(u) = {u}  ∪  neighbors(u)
```

Node `u`, when added to S, *covers itself* (by membership) and *all its neighbours*  
(one hop away). The minimum dominating set is the smallest collection of nodes  
whose coverage sets collectively cover the entire graph."""

CODE_PATH = """\
# ── Example 1: Path graph A-B-C-D-E ──────────────────────────────
# Who can cover everything in a path of 5 nodes?

path = Graph(edges=[('A','B'), ('B','C'), ('C','D'), ('D','E')])

print('Coverage of each node:')
for n in path.nodes():
    cov = set(path.neighbors(n)) | {n}
    print(f'  coverage({n}) = {sorted(cov)}')

print()
solutions = path.minimum_dominating_sets()
print(f'Minimum dominating sets (size {len(solutions[0])}):')
for s in format_solutions(solutions, sep=', '):
    print(f'  {{ {s} }}')

# Draw each solution
for i, sol in enumerate(solutions, 1):
    draw_mds(path, sol, title='Path graph A-B-C-D-E', solution_index=i)"""

MD_LOCS = """\
---
## The Sensor Network (`locaties`)

The original use case: sensors placed at locations connected by paths.  
An edge means the two sensors are within direct radio range.

The network has **two disconnected components** — two separate islands.  
Any dominating set must include at least one relay from each island."""

CODE_LOCS = """\
# ── locaties sensor network ───────────────────────────────────────
locaties = Graph(edges=[
    ('R', 'Q'), ('U', 'Z'), ('K', 'Y'), ('Q', 'Z'), ('L', 'K')
])

print('Nodes      :', locaties.nodes())
print('Connected? :', locaties.is_connected(), ' <- two separate islands!')
print('Components :', locaties.connected_components())
print()

solutions = locaties.minimum_dominating_sets()
print(f'Minimum relay stations needed : {len(solutions[0])}')
print(f'Number of optimal solutions   : {len(solutions)}')
print()
print('All optimal relay sets:')
for s in format_solutions(solutions, sep=', '):
    print(f'  {{ {s} }}')

# Draw all solutions
for i, sol in enumerate(solutions, 1):
    draw_mds(locaties, sol,
             title='Sensor network (locaties)',
             solution_index=i)"""

CODE_STAR = """\
# ── Example: star graph (one perfect dominator) ───────────────────
# A hub connected to 5 leaves: the hub alone is a dominating set.

star = Graph(edges=[
    ('hub','A'), ('hub','B'), ('hub','C'), ('hub','D'), ('hub','E')
])

solutions = star.minimum_dominating_sets()
print(f'Star graph — minimum dominating set size: {len(solutions[0])}')
print('Optimal sets:', format_solutions(solutions, sep=', '))

draw_mds(star, solutions[0], title='Star graph — hub covers everyone')"""

CODE_COMPLEX = """\
# ── Example: denser graph with more options ───────────────────────
# A grid-like structure — shows why multiple optimal solutions exist.

g = Graph(edges=[
    ('A','B'), ('A','D'),
    ('B','C'), ('B','E'),
    ('C','F'),
    ('D','E'), ('D','G'),
    ('E','F'), ('E','H'),
    ('F','I'),
    ('G','H'),
    ('H','I'),
])

print('Nodes:', g.nodes())
print('Edges:', g.edges())
solutions = g.minimum_dominating_sets()
print(f'\\nMinimum size: {len(solutions[0])}')
print(f'Number of optimal solutions: {len(solutions)}')
print()
for s in format_solutions(solutions, sep=', '):
    print(f'  {{ {s} }}')

# Draw the first two solutions side by side
for i, sol in enumerate(solutions[:3], 1):
    draw_mds(g, sol,
             title='9-node graph',
             solution_index=i)"""

MD_ALGO = """\
---
## The Backtracking Algorithm

`minimum_dominating_sets()` explores a **binary decision tree**:

```
For each node (in sorted order):
    Branch A → INCLUDE in S   (expand covered set)
    Branch B → SKIP            (covered set unchanged)
```

This gives 2ⁿ leaf branches — exponential in n. Two pruning rules cut  
branches that can never improve the current best:

### Pruning 1 — size bound
```
if |chosen| >= best_size_found_so_far:
    abandon this branch  (can only get worse)
```

### Pruning 2 — feasibility check
```
for every still-uncovered node v:
    if no remaining candidate can ever cover v:
        abandon this branch  (dead end)
```

The `debug=True` flag prints every decision and pruning event  
so you can watch the algorithm navigate the search tree."""

CODE_DEBUG = """\
# ── Debug walk-through on a tiny Graph ───────────────────────────
# Watch the backtracking explore and prune the search tree.

print('Graph: A-B-C-D  (path of 4 nodes)')
print('coverage(B) = {A,B,C}  and  coverage(C) = {B,C,D}')
print('So {B,C} or {B,D} or {A,C} or {A,D} should all work.')
print()

tiny = Graph(edges=[('A','B'), ('B','C'), ('C','D')], debug=True)
print()
solutions = tiny.minimum_dominating_sets()
print()
print('Verified solutions:', format_solutions(solutions, sep=', '))

for i, sol in enumerate(solutions, 1):
    draw_mds(tiny, sol,
             title='Path graph A-B-C-D',
             solution_index=i)"""

MD_SUMMARY = """\
---
## Summary

| Concept | Key insight |
|---|---|
| `_adj` | symmetric adjacency list — each edge stored in BOTH directions |
| `coverage(u)` | `{u}` + all neighbours of `u` |
| Dominating set | every node outside S is ≤ 1 hop from someone in S |
| Minimum dominating set | NP-complete — no polynomial algorithm known |
| Backtracking | explores 2ⁿ branches, pruned by size + feasibility checks |
| Multiple solutions | all equally-small sets are returned |

**When do multiple optimal solutions exist?**  
When two or more nodes have *symmetric* positions in the graph (like  
`R` and `Q` in the locaties path `R-Q-Z-U` — either can serve as the  
relay for that component's middle section)."""

# ================================================================== assemble

cells = [
    md(TITLE,       "md-00"),
    code(IMPORTS,   "co-01"),
    code(DRAW_FN,   "co-02"),
    md(MD_GRAPH,    "md-03"),
    code(CODE_GRAPH,"co-04"),
    md(MD_DOM,      "md-05"),
    code(CODE_PATH, "co-06"),
    md(MD_LOCS,     "md-07"),
    code(CODE_LOCS, "co-08"),
    code(CODE_STAR, "co-09"),
    code(CODE_COMPLEX,"co-10"),
    md(MD_ALGO,     "md-11"),
    code(CODE_DEBUG,"co-12"),
    md(MD_SUMMARY,  "md-13"),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3",
                       "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": cells,
}

with open('dominating_sets.ipynb', 'w', encoding='utf-8') as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print('dominating_sets.ipynb written.')

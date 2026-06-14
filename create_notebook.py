"""
create_notebook.py
Generates dag_problems.ipynb programmatically using json.dump.
Run:  python create_notebook.py
"""
import json

# ------------------------------------------------------------------ helpers

def code(src, cid):
    return {"cell_type": "code", "execution_count": None,
            "id": cid, "metadata": {}, "outputs": [], "source": src}

def md(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

# ================================================================== sources

TITLE = """\
# DAG — Directed Acyclic Graph: five toy problems

The `DAG` class is defined **inline** in the next cell.  
No external files are imported; only plain Python built-ins are used.

| # | Problem | Key method | Algorithm |
|---|---|---|---|
| 1 | **Task Scheduling** | `topological_sort()` | iterative — Kahn's BFS |
| 2 | **Critical Path** | `longest_path()` | iterative — DP over topo order |
| 3 | **Change Propagation** | `ancestors()` | recursive — DFS on reverse graph |
| 4 | **Fibonacci DP** | `spanning_tree()` + `in_degree()` | recursive — DFS |
| 5 | **Network Coverage** | `minimum_dominating_sets()` | recursive — backtracking |

---"""

IMPORTS = """\
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
plt.rcParams['figure.dpi'] = 110"""

DAG_CLASS = '''\
# ======================================================================
# format_solutions  —  module-level utility
# ======================================================================

def format_solutions(solutions, sep=':'):
    """
    Convert a list of solutions into human-readable strings.
    Each solution is a tuple/list of node names joined by sep.

    >>> format_solutions([("A","B","C"),("A","C")], sep=":")
    ['A:B:C', 'A:C']
    """
    return [sep.join(str(n) for n in sol) for sol in solutions]


# ======================================================================
# DAG  —  Directed Acyclic Graph
#
# Internal state
#   _forward_adj : dict[str, list[str]]   node -> list of children
#   _reverse_adj : dict[str, list[str]]   node -> list of parents
#
# Sections
#   A  CONSTRUCTION   __init__ and classmethods
#   B  PRIVATE        internal helpers (prefixed _)
#   C  AUXILIARY      simple accessors; no recursion; O(1) or O(degree)
#   D  CORE           algorithms; recursion type annotated per method
# ======================================================================

class DAG:

    # ==================================================================
    # A. CONSTRUCTION
    # ==================================================================

    def __init__(self, edges=None, nodes=None):
        """
        Primary constructor.
        edges : list of (src, dst) tuples
        nodes : optional list of str  (to include isolated nodes)
        Raises ValueError if the supplied edges form a cycle.
        """
        self._forward_adj = {}
        self._reverse_adj = {}
        if nodes:
            for n in nodes:
                self._ensure_node(n)
        if edges:
            for src, dst in edges:
                self._add_edge(src, dst)
        if not self._check_acyclic():
            raise ValueError('Edges form a cycle — not a valid DAG.')

    @classmethod
    def from_adjacency_dict(cls, adj):
        """Build from {node: [child, ...]} adjacency dict."""
        nodes = list(adj.keys())
        edges = [(s, d) for s, cs in adj.items() for d in cs]
        return cls(edges=edges, nodes=nodes)

    @classmethod
    def from_nodes_and_edges(cls, nodes, edges):
        """Build from explicit node list + edge list. Isolated nodes preserved."""
        return cls(edges=edges, nodes=nodes)

    # ==================================================================
    # B. PRIVATE HELPERS
    # ==================================================================

    def _ensure_node(self, n):
        """Register node n if it does not yet exist."""
        if n not in self._forward_adj:
            self._forward_adj[n] = []
            self._reverse_adj[n] = []

    def _add_edge(self, src, dst):
        """Add directed edge src->dst; keeps both adjacency dicts in sync."""
        self._ensure_node(src)
        self._ensure_node(dst)
        if dst not in self._forward_adj[src]:
            self._forward_adj[src].append(dst)
        if src not in self._reverse_adj[dst]:
            self._reverse_adj[dst].append(src)

    def _check_acyclic(self):
        """
        RECURSIVE — simple DFS  (three-colour algorithm).
        WHITE=0 unvisited  |  GRAY=1 on stack  |  BLACK=2 done.
        Reaching a GRAY node is a back-edge, which signals a cycle.
        Returns True when the graph is acyclic.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._forward_adj}

        def dfs(u):
            color[u] = GRAY
            for v in self._forward_adj[u]:
                if color[v] == GRAY:
                    return False               # back-edge => cycle
                if color[v] == WHITE and not dfs(v):
                    return False
            color[u] = BLACK
            return True

        return all(dfs(n) for n in list(self._forward_adj) if color[n] == WHITE)

    # ==================================================================
    # C. AUXILIARY — simple accessors, no recursion
    # ==================================================================

    def nodes(self):
        """Sorted list of all node names."""
        return sorted(self._forward_adj.keys())

    def edges(self):
        """Sorted list of all (src, dst) edge tuples."""
        return [(s, d)
                for s in sorted(self._forward_adj)
                for d in self._forward_adj[s]]

    def children(self, node):
        """Direct successors — nodes this node points TO."""
        return list(self._forward_adj.get(node, []))

    def parents(self, node):
        """Direct predecessors — nodes that point TO this node."""
        return list(self._reverse_adj.get(node, []))

    def in_degree(self, node):
        """Number of incoming edges (= number of parents)."""
        return len(self._reverse_adj.get(node, []))

    def out_degree(self, node):
        """Number of outgoing edges (= number of children)."""
        return len(self._forward_adj.get(node, []))

    def roots(self):
        """Nodes with no incoming edges (sources / entry points)."""
        return sorted(n for n in self._forward_adj if not self._reverse_adj[n])

    def leaves(self):
        """Nodes with no outgoing edges (sinks / exit points)."""
        return sorted(n for n in self._forward_adj if not self._forward_adj[n])

    def __repr__(self):
        return f'DAG(nodes={self.nodes()}, edges={self.edges()})'

    def __str__(self):
        lines = ['DAG:']
        for node in self.nodes():
            ch = self.children(node)
            pa = self.parents(node)
            lines.append(f'  {node}  parents={pa or []}  children={ch or []}')
        return '\\n'.join(lines)

    # ==================================================================
    # D. CORE ALGORITHMS
    # ==================================================================
    # D1 Reachability    D2 Ordering    D3 Path optimisation
    # D4 Tree extraction    D5 Set covering

    # ------------------------------------------------------------------
    # D1. Reachability
    # ------------------------------------------------------------------

    def has_path(self, src, dst):
        """
        RECURSIVE — simple DFS.
        True if a directed path from src to dst exists.
        Short-circuits on first discovery of dst.
        """
        if src == dst:
            return True
        visited = set()

        def dfs(u):
            if u == dst:
                return True
            visited.add(u)
            return any(dfs(v) for v in self._forward_adj.get(u, [])
                       if v not in visited)

        return dfs(src)

    def all_paths(self, src, dst):
        """
        RECURSIVE — DFS with backtracking.
        Returns every directed path src->dst as a list of node lists.

        The path list grows on each recursive descent and is popped on
        return — classic generate-and-test backtracking.
        Warning: path count can be exponential.
        """
        result = []

        def dfs(u, path):
            if u == dst:
                result.append(list(path))
                return
            for v in self._forward_adj.get(u, []):
                if v not in path:
                    path.append(v)
                    dfs(v, path)
                    path.pop()          # backtrack

        dfs(src, [src])
        return result

    def descendants(self, node):
        """
        RECURSIVE — simple DFS (forward graph).
        All nodes reachable FROM node (node itself excluded).
        """
        visited = set()

        def dfs(u):
            for v in self._forward_adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dfs(v)

        dfs(node)
        return sorted(visited)

    def ancestors(self, node):
        """
        RECURSIVE — simple DFS (reverse graph).
        All nodes from which node is reachable (node itself excluded).
        Walks _reverse_adj to travel upstream.
        """
        visited = set()

        def dfs(u):
            for v in self._reverse_adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dfs(v)

        dfs(node)
        return sorted(visited)

    # ------------------------------------------------------------------
    # D2. Ordering
    # ------------------------------------------------------------------

    def topological_sort(self):
        """
        ITERATIVE — Kahn's algorithm (BFS over in-degree counts).
        Every node appears after all its predecessors.
        Alphabetical tie-breaking for a deterministic result.
        """
        in_deg = {n: self.in_degree(n) for n in self._forward_adj}
        queue  = sorted(n for n in in_deg if in_deg[n] == 0)
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in sorted(self._forward_adj[node]):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
                    queue.sort()
        return result

    # ------------------------------------------------------------------
    # D3. Path optimisation
    # ------------------------------------------------------------------

    def shortest_path(self, src, dst):
        """
        ITERATIVE — BFS (list as FIFO queue).
        Path with fewest edges from src to dst, or [] if no path.
        BFS guarantees first arrival at dst is via the shortest route.
        """
        if src not in self._forward_adj or dst not in self._forward_adj:
            return []
        queue, visited = [[src]], {src}
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == dst:
                return path
            for child in self._forward_adj.get(node, []):
                if child not in visited:
                    visited.add(child)
                    queue.append(path + [child])
        return []

    def longest_path(self, src, dst):
        """
        ITERATIVE — DP over topological order, O(V + E).
        Path with most edges from src to dst, or [] if no path.

        dist[v] = max edge-count from src to v  (-1 = unreachable).
        prev[v] = predecessor on the best path to v.
        Topological order ensures dist[u] is final before relaxing
        u's outgoing edges.
        """
        topo = self.topological_sort()
        dist = {n: -1   for n in topo}
        prev = {n: None for n in topo}
        dist[src] = 0
        for u in topo:
            if dist[u] == -1:
                continue
            for v in self._forward_adj.get(u, []):
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1
                    prev[v] = u
        if dist.get(dst, -1) == -1:
            return []
        path, cur = [], dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path[::-1]

    # ------------------------------------------------------------------
    # D4. Tree extraction
    # ------------------------------------------------------------------

    def spanning_tree(self, root=None):
        """
        RECURSIVE — simple DFS.
        Returns {node: [children_in_tree]} rooted at root.

        First-discovery DFS: the first edge that reaches a node becomes
        a tree edge; subsequent edges to the same node are cross-edges
        and are dropped.  This collapses the DAG into a tree.
        """
        if root is None:
            roots = self.roots()
            if not roots:
                raise ValueError('DAG has no root.')
            root = roots[0]
        tree, visited = {root: []}, {root}

        def dfs(u):
            for v in self._forward_adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    tree[u].append(v)
                    tree[v] = []
                    dfs(v)

        dfs(root)
        return tree

    # ------------------------------------------------------------------
    # D5. Set covering
    # ------------------------------------------------------------------

    def minimum_dominating_sets(self, directed=True):
        """
        RECURSIVE — backtracking with size pruning + feasibility pruning.

        Find all minimum dominating sets S:
          every node NOT in S must be reachable from some node in S
          in exactly one hop.

        Equivalent to Minimum Set Cover  (NP-complete in general).

        Parameters
        ----------
        directed : bool, default True
            True  — directed domination:
                      coverage(u) = {u} union children(u)
            False — undirected domination (both edge directions count):
                      coverage(u) = {u} union children(u) union parents(u)
                    Use this when edges represent bidirectional connections
                    stored as one-direction tuples.

        Pruning 1 — size:  |chosen| >= best found  =>  abandon branch.
        Pruning 2 — feasibility:  some uncovered node v has no remaining
                    candidate that can still cover it  =>  dead branch.

        Returns a list of optimal solutions (each a sorted list of nodes).
        """
        nodes     = self.nodes()
        n         = len(nodes)
        all_nodes = set(nodes)

        if directed:
            coverage = {u: set(self.children(u)) | {u} for u in nodes}
        else:
            coverage = {u: set(self.children(u)) | set(self.parents(u)) | {u}
                        for u in nodes}
        node_idx    = {nd: i for i, nd in enumerate(nodes)}
        coverers_of = {v: set() for v in nodes}
        for u in nodes:
            for v in coverage[u]:
                coverers_of[v].add(node_idx[u])

        best_size, results = [n], []

        def backtrack(idx, chosen, covered):
            if covered == all_nodes:
                size = len(chosen)
                if size < best_size[0]:
                    best_size[0] = size
                    results.clear()
                    results.append(tuple(chosen))
                elif size == best_size[0]:
                    results.append(tuple(chosen))
                return
            if len(chosen) >= best_size[0]:              # pruning 1
                return
            for v in all_nodes - covered:                # pruning 2
                if not any(i >= idx for i in coverers_of[v]):
                    return
            if idx >= n:
                return
            node = nodes[idx]
            chosen.append(node)                          # branch A: include
            backtrack(idx + 1, chosen, covered | coverage[node])
            chosen.pop()
            backtrack(idx + 1, chosen, covered)          # branch B: skip

        backtrack(0, [], set())
        return [sorted(r) for r in results]


print('DAG loaded.  Sections:  A Construction  |  B Private  |  C Auxiliary  |  D Core')
'''

DRAW_DAG = """\
CLR_NORMAL    = '#3A86D4'   # blue  — default nodes / edges
CLR_HIGHLIGHT = '#D94F4F'   # red   — highlighted nodes / edges
CLR_EDGE      = '#AAAAAA'   # grey  — default edge


def draw_dag(dag, title='DAG',
             highlight_nodes=None,
             highlight_edges=None,
             legend=None):
    \"\"\"
    Visualise a DAG with a hierarchical left-to-right layered layout.

    Layout rule
    -----------
    Each node's x-position equals the length of the longest path from
    any root to that node, so cause always appears left of effect.

    Parameters
    ----------
    dag             : DAG instance
    title           : figure title string
    highlight_nodes : list of node names to draw in red
    highlight_edges : list of (src, dst) tuples to draw in red
    legend          : list of (colour, label) tuples for a legend box
    \"\"\"
    highlight_nodes = set(highlight_nodes or [])
    highlight_edges = set(tuple(e) for e in (highlight_edges or []))

    nodes = dag.nodes()
    if not nodes:
        print('(empty graph)')
        return

    # ── Layered layout ────────────────────────────────────────────────
    topo  = dag.topological_sort()
    layer = {n: 0 for n in nodes}
    for n in topo:
        for child in dag.children(n):
            layer[child] = max(layer[child], layer[n] + 1)

    buckets = {}
    for n, l in layer.items():
        buckets.setdefault(l, []).append(n)
    for l in buckets:
        buckets[l].sort()

    num_layers   = max(buckets) + 1
    max_in_layer = max(len(v) for v in buckets.values())

    X_STEP, Y_STEP, NODE_R = 3.2, 2.0, 0.55

    pos = {}
    for l, ns in buckets.items():
        for i, n in enumerate(ns):
            pos[n] = (l * X_STEP, -(i - (len(ns) - 1) / 2.0) * Y_STEP)

    fig_w = max(7, num_layers * X_STEP + 2)
    fig_h = max(4, max_in_layer * Y_STEP + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')

    # ── Edges ─────────────────────────────────────────────────────────
    for src, dst in dag.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        dx, dy  = x1 - x0, y1 - y0
        length  = (dx**2 + dy**2) ** 0.5
        if length < 1e-9:
            continue
        ux, uy  = dx / length, dy / length
        ex0, ey0 = x0 + ux * NODE_R, y0 + uy * NODE_R
        ex1, ey1 = x1 - ux * NODE_R, y1 - uy * NODE_R
        is_hi  = (src, dst) in highlight_edges
        ax.annotate('', xy=(ex1, ey1), xytext=(ex0, ey0),
                    arrowprops=dict(
                        arrowstyle='-|>',
                        color=CLR_HIGHLIGHT if is_hi else CLR_EDGE,
                        lw=2.8 if is_hi else 1.5,
                        mutation_scale=14))

    # ── Nodes ─────────────────────────────────────────────────────────
    for n in nodes:
        x, y = pos[n]
        fc   = CLR_HIGHLIGHT if n in highlight_nodes else CLR_NORMAL
        ax.add_patch(plt.Circle((x, y), NODE_R, color=fc, ec='white', lw=2, zorder=3))
        ax.text(x, y, n if len(n) <= 7 else n[:6] + '…',
                ha='center', va='center',
                fontsize=9, color='white', fontweight='bold', zorder=4)

    # ── Legend ────────────────────────────────────────────────────────
    if legend:
        ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in legend],
                  loc='lower right', fontsize=9, framealpha=0.85)

    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    pad = 1.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    plt.tight_layout()
    plt.show()


print('draw_dag ready.')"""

MD_P1 = """\
---
## Problem 1 — Task Scheduling

A software project has tasks with dependency constraints.  
An edge **X → Y** means *"X must be completed before Y can start"*.

**Question:** in what order can a single developer execute all tasks?  
**Answer:** any **topological sort** of the dependency DAG is a valid schedule.

`topological_sort()` uses **Kahn's iterative BFS-over-in-degrees** algorithm and
runs in O(V + E).  The result is not unique in general — every valid linear
extension of the partial order is a correct answer."""

CODE_P1 = """\
g_sched = DAG(edges=[
    ('design',   'backend'),
    ('design',   'frontend'),
    ('backend',  'tests'),
    ('frontend', 'tests'),
    ('tests',    'review'),
    ('review',   'deploy'),
    ('tests',    'deploy'),
])

order = g_sched.topological_sort()

print('=== Task Scheduling ===')
print('Valid order:', ' → '.join(order))
print()
print('Checking every task appears after all its dependencies:')
for i, task in enumerate(order):
    deps = g_sched.parents(task)
    if deps:
        for d in deps:
            assert order.index(d) < i
        print(f'  {task:10s} ← after {deps}  ✓')
    else:
        print(f'  {task:10s} ← no dependencies (can start immediately)  ✓')

draw_dag(g_sched, title='Problem 1: Task Scheduling')"""

MD_P2 = """\
---
## Problem 2 — Critical Path Method (CPM)

Each task has a duration in days.  
The **minimum project duration** equals the weight of the **longest path**
from `START` to `END` — the *critical path*.

Delaying any task on the critical path delays the whole project.  
Tasks off it have *slack* and can slip without impact.

`longest_path()` uses **DP over topological order** in O(V + E)."""

CODE_P2 = """\
durations = {
    'design': 3, 'backend': 5, 'frontend': 4,
    'tests': 2, 'review': 1, 'deploy': 1,
}

g_cpm = DAG(edges=[
    ('START',    'design'),
    ('design',   'backend'),
    ('design',   'frontend'),
    ('backend',  'tests'),
    ('frontend', 'tests'),
    ('tests',    'review'),
    ('review',   'END'),
    ('tests',    'END'),
])

crit  = g_cpm.longest_path('START', 'END')
c_edges = [(crit[i], crit[i+1]) for i in range(len(crit) - 1)]
total   = sum(durations.get(n, 0) for n in crit)

print('=== Critical Path Method ===')
print('Critical path :', ' → '.join(crit))
print(f'Min duration  : {total} days')
print()
for n in crit:
    d = durations.get(n, 0)
    print(f'  {n:10s}  {"█" * d} {d}d')

draw_dag(g_cpm,
         title='Problem 2: Critical Path (red = bottleneck)',
         highlight_nodes=[n for n in crit if n not in ('START', 'END')],
         highlight_edges=c_edges,
         legend=[(CLR_NORMAL,    'task with slack'),
                 (CLR_HIGHLIGHT, 'critical path')])"""

MD_P3 = """\
---
## Problem 3 — Change Propagation in a Codebase

An edge **X → Y** means *"module X directly imports module Y"*.

Modifying module Y means every module that (transitively) depends on Y
might be broken and must be re-tested.

- **Direct importers** = `parents(Y)` — one hop\n- **Transitive importers** = `ancestors(Y)` — all hops

`ancestors()` uses **recursive DFS on the reverse graph** (_reverse_adj)."""

CODE_P3 = """\
g_code = DAG.from_adjacency_dict({
    'app':   ['ui', 'api'],
    'ui':    ['utils', 'theme'],
    'api':   ['utils', 'db'],
    'utils': [],
    'theme': [],
    'db':    [],
})

changed    = 'utils'
direct     = g_code.parents(changed)
transitive = g_code.ancestors(changed)

print('=== Change Propagation ===')
print(f'Modified module      : {changed}')
print(f'Direct importers     : {direct}')
print(f'Transitive importers : {transitive}')
print(f'utils itself uses    : {g_code.descendants(changed)} (unchanged)')

affected_edges = [(s, d) for s, d in g_code.edges()
                  if s in transitive or d in transitive
                  or s == changed or d == changed]

draw_dag(g_code,
         title=f"Problem 3: Change Propagation — '{changed}' was modified",
         highlight_nodes=transitive + [changed],
         highlight_edges=affected_edges,
         legend=[(CLR_NORMAL,    'unaffected module'),
                 (CLR_HIGHLIGHT, 'modified / must re-test')])"""

MD_P4 = """\
---
## Problem 4 — Fibonacci: Tree vs DAG (Memoization)

Without memoization, `fib(n)` explores an exponential *call tree*.  
With memoization, each unique subproblem is computed once — the
structure is a **DAG** because some nodes have **multiple parents**.

`spanning_tree()` shows which nodes a memoized DFS *first visits*  
(subsequent edges to the same node become cross-edges and are dropped).

Nodes with `in_degree > 1` are *shared subproblems* — they would be  
recomputed multiple times by a naive recursive implementation."""

CODE_P4 = """\
g_fib = DAG(edges=[
    ('f5', 'f4'), ('f5', 'f3'),
    ('f4', 'f3'), ('f4', 'f2'),
    ('f3', 'f2'), ('f3', 'f1'),
    ('f2', 'f1'), ('f2', 'f0'),
])

shared = [n for n in g_fib.nodes() if g_fib.in_degree(n) > 1]
tree   = g_fib.spanning_tree('f5')

print('=== Fibonacci Subproblem DAG ===')
print('Shared subproblems (in-degree > 1):', shared)
print()
print('Spanning tree (= memoized evaluation order from f5):')
for node, children in sorted(tree.items()):
    arrow = children if children else '(base case)'
    print(f'  {node} -> {arrow}')
print()
print('In-degree per node:')
for n in g_fib.topological_sort():
    deg  = g_fib.in_degree(n)
    note = '  <- shared subproblem!' if deg > 1 else ''
    print(f'  {n}: in-degree = {deg}{note}')

draw_dag(g_fib,
         title='Problem 4: Fibonacci DAG  (red = shared / memoized nodes)',
         highlight_nodes=shared,
         legend=[(CLR_NORMAL,    'unique subproblem'),
                 (CLR_HIGHLIGHT, 'shared subproblem (saved by memoization)')])"""

MD_P5 = """\
---
## Problem 5 — Minimum Network Coverage (Dominating Set)

A wireless sensor network: edge **A → B** means A can relay directly to B.  
We want the **fewest relay stations** (subset S) such that every sensor
outside S is exactly **one hop** from a relay.

This is the **directed Minimum Dominating Set** problem,
equivalent to **Minimum Set Cover** (NP-complete in general).

`minimum_dominating_sets()` finds *all* optimal solutions with  
**recursive backtracking + size pruning + feasibility pruning**."""

CODE_P5 = """\
g_net = DAG(edges=[
    ('hub', 'A'), ('hub', 'B'),
    ('A',   'C'), ('A',   'D'),
    ('B',   'D'), ('B',   'E'),
    ('C',   'F'),
    ('D',   'F'), ('D',   'G'),
    ('E',   'G'),
])

# directed=True (default): edge A->B means only A covers B, not the reverse
solutions_dir = g_net.minimum_dominating_sets(directed=True)

print('=== Minimum Network Coverage ===')
print(f'Nodes : {g_net.nodes()}')
print(f'Edges : {g_net.edges()}')
print()
print('directed=True  (only outgoing edges count for coverage):')
print(f'  size={len(solutions_dir[0])}   sets={format_solutions(solutions_dir, sep=",")}')

# directed=False: treat edges as bidirectional — A-B means A covers B and B covers A
solutions_undir = g_net.minimum_dominating_sets(directed=False)
print()
print('directed=False (both edge directions count — undirected domination):')
print(f'  size={len(solutions_undir[0])}   sets={format_solutions(solutions_undir, sep=",")}')

# ── Bonus: the locaties example the user raised ──────────────────────
print()
print('--- locaties (undirected roads) ---')
locaties = DAG(edges=[
    ('R', 'Q'), ('U', 'Z'), ('K', 'Y'), ('Q', 'Z'), ('L', 'K')
])
sol = locaties.minimum_dominating_sets(directed=False)
print(f'Solutions: {format_solutions(sol, sep=",")}')
print('Expected : [K,Q,U]  [K,Q,Z]  [K,R,U]  [K,R,Z]')

best        = solutions_undir[0]
relay_edges = [(u, v) for u, v in g_net.edges()
               if u in best or v in best]

draw_dag(g_net,
         title='Problem 5: Minimum Network Coverage  directed=False  (red = relay stations)',
         highlight_nodes=best,
         highlight_edges=relay_edges,
         legend=[(CLR_NORMAL,    'covered sensor (1 hop from relay)'),
                 (CLR_HIGHLIGHT, 'relay station')])"""

MD_SUMMARY = """\
---
## Summary

| Problem | Concept | Algorithm |
|---|---|---|
| Task scheduling | Topological sort | Iterative — Kahn's BFS, O(V+E) |
| Critical path | Longest path | Iterative — DP over topo order, O(V+E) |
| Change propagation | Ancestors / descendants | Recursive — DFS, O(V+E) |
| Fibonacci DP | Shared subproblems | Recursive — DFS, O(V+E) |
| Network coverage | Minimum dominating set | Recursive — backtracking, NP-complete |

Every problem reduces to **walking the DAG** in a specific way —
confirming that the recursive structure of a problem is always
mirrored by the structure of a graph."""

# ================================================================== assemble

cells = [
    md(TITLE,     "md-00"),
    code(IMPORTS, "co-01"),
    code(DAG_CLASS, "co-02"),
    code(DRAW_DAG,  "co-03"),
    md(MD_P1,     "md-04"),
    code(CODE_P1, "co-05"),
    md(MD_P2,     "md-06"),
    code(CODE_P2, "co-07"),
    md(MD_P3,     "md-08"),
    code(CODE_P3, "co-09"),
    md(MD_P4,     "md-10"),
    code(CODE_P4, "co-11"),
    md(MD_P5,     "md-12"),
    code(CODE_P5, "co-13"),
    md(MD_SUMMARY, "md-14"),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.0"
        }
    },
    "cells": cells,
}

with open('dag_problems.ipynb', 'w', encoding='utf-8') as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)

print('dag_problems.ipynb written successfully.')

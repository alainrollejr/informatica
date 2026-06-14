"""
dag.py — A generic Directed Acyclic Graph (DAG) implementation.

Uses only plain Python built-ins: dicts, lists, tuples, sets.
No external or standard-library imports.

A DAG is a directed graph with no directed cycles.  It generalises trees:
  - every tree is a DAG  (each non-root node has exactly one parent)
  - a DAG allows a node to have multiple parents

Internal adjacency naming
-------------------------
_forward_adj : dict[str, list[str]]   node -> list of children
_reverse_adj : dict[str, list[str]]   node -> list of parents

Methods are organised in four sections
---------------------------------------
A  CONSTRUCTION   — __init__ and classmethods
B  PRIVATE        — internal helpers (prefixed _)
C  AUXILIARY      — simple accessors; no recursion, O(1) or O(degree)
D  CORE           — non-trivial algorithms (recursion type annotated)
"""


# ====================================================================== #
#  Module-level utility                                                   #
# ====================================================================== #

def format_solutions(solutions, sep=":"):
    """
    Convert a list of solutions into human-readable strings.

    Each solution is a tuple or list of node names joined by `sep`.

    Parameters
    ----------
    solutions : list of tuple/list   e.g. [("A","B","C"), ("A","C")]
    sep       : str                  separator between nodes (default ":")

    Returns
    -------
    list of str   e.g. ["A:B:C", "A:C"]

    Examples
    --------
    >>> format_solutions([("A","B","C"), ("A","C")], sep=":")
    ['A:B:C', 'A:C']
    >>> format_solutions([("A","B","C"), ("A","C")], sep=",")
    ['A,B,C', 'A,C']
    >>> format_solutions([("A","B","C"), ("A","C")], sep=" -> ")
    ['A -> B -> C', 'A -> C']
    """
    return [sep.join(str(n) for n in sol) for sol in solutions]


# ====================================================================== #
#  DAG class                                                              #
# ====================================================================== #

class DAG:
    """
    Directed Acyclic Graph.

    Internal state
    --------------
    _forward_adj : dict[str, list[str]]  — node -> list of children
    _reverse_adj : dict[str, list[str]]  — node -> list of parents

    Both structures are kept in sync on every mutation so that parent
    lookups cost O(degree) rather than O(E).
    """

    # ==================================================================
    # A. CONSTRUCTION
    # ==================================================================

    def __init__(self, edges=None, nodes=None):
        """
        Primary constructor.

        Parameters
        ----------
        edges : list of (src, dst) tuples  — directed edges to add
        nodes : list of str (optional)     — explicit node names;
                useful for isolated nodes that appear in no edge

        Raises ValueError if the supplied edges form a cycle.

        Example
        -------
        >>> g = DAG(edges=[("A","B"), ("A","C"), ("B","D")])
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
            raise ValueError(
                "The supplied edges contain a cycle — not a valid DAG.")

    @classmethod
    def from_adjacency_dict(cls, adj):
        """
        Build a DAG from an adjacency dictionary {node: [child, ...]}.

        Nodes that appear only as children are created automatically.

        Example
        -------
        >>> g = DAG.from_adjacency_dict({"A": ["B","C"], "B": ["D"], "C": [], "D": []})
        """
        nodes = list(adj.keys())
        edges = []
        for src, children in adj.items():
            for dst in children:
                edges.append((src, dst))
        return cls(edges=edges, nodes=nodes)

    @classmethod
    def from_nodes_and_edges(cls, nodes, edges):
        """
        Build a DAG from an explicit node list and an edge list.

        Isolated nodes (not part of any edge) are preserved because
        they are listed in `nodes`.

        Example
        -------
        >>> g = DAG.from_nodes_and_edges(["A","B","C","Z"], [("A","B"),("B","C")])
        >>> "Z" in g.nodes()
        True
        """
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
        """Add directed edge src -> dst; keeps both adjacency dicts in sync."""
        self._ensure_node(src)
        self._ensure_node(dst)
        if dst not in self._forward_adj[src]:
            self._forward_adj[src].append(dst)
        if src not in self._reverse_adj[dst]:
            self._reverse_adj[dst].append(src)

    def _check_acyclic(self):
        """
        RECURSIVE — simple DFS (three-colour algorithm).

        Colours:
          WHITE = 0  not yet visited
          GRAY  = 1  currently on the DFS recursion stack
          BLACK = 2  fully explored

        A back-edge (reaching a GRAY node) signals a cycle.
        Returns True when the graph is acyclic.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._forward_adj}

        def dfs(u):
            color[u] = GRAY
            for v in self._forward_adj[u]:
                if color[v] == GRAY:
                    return False          # back-edge → cycle
                if color[v] == WHITE and not dfs(v):
                    return False
            color[u] = BLACK
            return True

        for node in self._forward_adj:
            if color[node] == WHITE:
                if not dfs(node):
                    return False
        return True

    # ==================================================================
    # C. AUXILIARY — simple accessors, no recursion
    # ==================================================================

    def nodes(self):
        """Return a sorted list of all node names in the DAG."""
        return sorted(self._forward_adj.keys())

    def edges(self):
        """Return a sorted list of all directed edges as (src, dst) tuples."""
        result = []
        for src in sorted(self._forward_adj):
            for dst in self._forward_adj[src]:
                result.append((src, dst))
        return result

    def children(self, node):
        """Direct successors of node — the nodes this node points TO."""
        return list(self._forward_adj.get(node, []))

    def parents(self, node):
        """Direct predecessors of node — the nodes that point TO this node."""
        return list(self._reverse_adj.get(node, []))

    def in_degree(self, node):
        """Number of incoming edges (= number of parents)."""
        return len(self._reverse_adj.get(node, []))

    def out_degree(self, node):
        """Number of outgoing edges (= number of children)."""
        return len(self._forward_adj.get(node, []))

    def roots(self):
        """
        Return nodes with no incoming edges (sources / entry points).
        A valid DAG always has at least one root.
        """
        return sorted(
            n for n in self._forward_adj if len(self._reverse_adj[n]) == 0)

    def leaves(self):
        """Return nodes with no outgoing edges (sinks / exit points)."""
        return sorted(
            n for n in self._forward_adj if len(self._forward_adj[n]) == 0)

    def __repr__(self):
        return f"DAG(nodes={self.nodes()}, edges={self.edges()})"

    def __str__(self):
        lines = ["DAG:"]
        for node in self.nodes():
            ch = self.children(node)
            pa = self.parents(node)
            lines.append(
                f"  {node}"
                f"  parents={pa if pa else '—'}"
                f"  children={ch if ch else '—'}"
            )
        return "\n".join(lines)

    # ==================================================================
    # D. CORE ALGORITHMS
    # ==================================================================

    # ------------------------------------------------------------------
    # D1. Reachability
    # ------------------------------------------------------------------

    def has_path(self, src, dst):
        """
        RECURSIVE — simple DFS.

        Return True if a directed path from src to dst exists.
        Short-circuits as soon as dst is found.
        """
        if src == dst:
            return True
        visited = set()

        def dfs(u):
            if u == dst:
                return True
            visited.add(u)
            for v in self._forward_adj.get(u, []):
                if v not in visited and dfs(v):
                    return True
            return False

        return dfs(src)

    def all_paths(self, src, dst):
        """
        RECURSIVE — DFS with backtracking.

        Return every directed path from src to dst as a list of node lists.
        The current path grows on the way down the recursion tree and is
        restored (backtracked) when returning — a classic generate-and-test
        pattern.

        Warning: the number of paths can be exponential in the graph size.

        Example
        -------
        >>> g.all_paths("A", "D")
        [['A', 'B', 'D'], ['A', 'C', 'D']]
        """
        result = []

        def dfs(u, path):
            if u == dst:
                result.append(list(path))   # record a complete path
                return
            for v in self._forward_adj.get(u, []):
                if v not in path:
                    path.append(v)
                    dfs(v, path)
                    path.pop()              # backtrack

        dfs(src, [src])
        return result

    def descendants(self, node):
        """
        RECURSIVE — simple DFS (forward graph).

        Return a sorted list of all nodes reachable FROM node
        (node itself is excluded).
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

        Return a sorted list of all nodes from which node is reachable
        (node itself is excluded).
        Walks _reverse_adj instead of _forward_adj to travel upstream.
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

        Return nodes in an order where every node appears only after
        all its predecessors have appeared.

        Steps:
          1. Enqueue all nodes whose in-degree is 0.
          2. Pop a node, append to result, decrement in-degree of each
             child; re-enqueue children whose in-degree drops to 0.
          3. Repeat until the queue is empty.

        Ties are broken alphabetically for a deterministic result.
        Topological order is not unique in general.
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
        ITERATIVE — BFS (plain list as FIFO queue).

        Return the path with the fewest edges from src to dst,
        or [] when no path exists.
        BFS guarantees the first time dst is reached is via the shortest path.
        """
        if src not in self._forward_adj or dst not in self._forward_adj:
            return []

        queue   = [[src]]
        visited = {src}

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

        Return the path with the most edges from src to dst,
        or [] when no path exists.

        dist[v] = max number of edges from src to v (-1 = unreachable).
        prev[v] = predecessor on the best known path to v.

        Processing nodes in topological order guarantees dist[u] is
        finalised before any outgoing edge of u is relaxed.
        """
        topo    = self.topological_sort()
        NEG_INF = -1

        dist = {n: NEG_INF for n in topo}
        prev = {n: None    for n in topo}
        dist[src] = 0

        for u in topo:
            if dist[u] == NEG_INF:
                continue
            for v in self._forward_adj.get(u, []):
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1
                    prev[v] = u

        if dist.get(dst, NEG_INF) == NEG_INF:
            return []

        # Reconstruct path by following prev[] pointers back from dst
        path = []
        cur  = dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # D4. Tree extraction
    # ------------------------------------------------------------------

    def spanning_tree(self, root=None):
        """
        RECURSIVE — simple DFS.

        Return a spanning tree rooted at `root` as {node: [children_in_tree]}.
        Defaults to the first alphabetical root when root is None.

        Each node is visited exactly once (first-discovery wins).
        Edges that would lead to an already-visited node become cross-edges
        and are dropped — this is precisely what collapses a DAG into a tree.
        """
        if root is None:
            candidates = self.roots()
            if not candidates:
                raise ValueError(
                    "DAG has no root (every node has at least one parent).")
            root = candidates[0]

        tree    = {root: []}
        visited = {root}

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
          every node NOT in S must be reachable from some node IN S
          in exactly one hop.

        This is the Minimum Set Cover / Dominating Set problem,
        NP-complete in general.

        Parameters
        ----------
        directed : bool, default True
            True  — only outgoing edges count for coverage:
                      coverage(u) = {u} ∪ children(u)
                    Use for directed problems (e.g. relay networks
                    with one-way transmission).
            False — both in- and out-edges count (undirected domination):
                      coverage(u) = {u} ∪ children(u) ∪ parents(u)
                    Use when edges represent bidirectional connections
                    stored as one-direction tuples in the DAG
                    (e.g. roads, proximity links, undirected graphs).

        Algorithm
        ---------
        Nodes are processed in a fixed sorted order.  At each index we
        choose to include the node (Branch A) or skip it (Branch B),
        then recurse.  Two pruning rules cut the search tree early:

          Pruning 1 — size:
            If |chosen| >= best size found so far, abandon this branch.

          Pruning 2 — feasibility:
            For every still-uncovered node v, at least one remaining
            candidate (index >= current) must be able to cover it.
            If not, the branch is a dead end.

        Returns a list of optimal solutions, each sorted list of node names.
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

        best_size = [n]
        results   = []

        def backtrack(idx, chosen, covered):
            # ── Base case ────────────────────────────────────────────
            if covered == all_nodes:
                size = len(chosen)
                if size < best_size[0]:
                    best_size[0] = size
                    results.clear()
                    results.append(tuple(chosen))
                elif size == best_size[0]:
                    results.append(tuple(chosen))
                return

            # ── Pruning 1: size ──────────────────────────────────────
            if len(chosen) >= best_size[0]:
                return

            # ── Pruning 2: feasibility ───────────────────────────────
            for v in all_nodes - covered:
                if not any(i >= idx for i in coverers_of[v]):
                    return    # v can never be covered → dead branch

            if idx >= n:
                return

            node = nodes[idx]
            # Branch A: include nodes[idx]
            chosen.append(node)
            backtrack(idx + 1, chosen, covered | coverage[node])
            chosen.pop()
            # Branch B: skip nodes[idx]
            backtrack(idx + 1, chosen, covered)

        backtrack(0, [], set())
        return [sorted(r) for r in results]


# ====================================================================== #
#  Standalone demo                                                        #
# ====================================================================== #

if __name__ == "__main__":

    SEP = "=" * 60

    print(SEP)
    print("EXAMPLE 1 — Simple pipeline")
    print(SEP)
    g1 = DAG(edges=[("A","B"), ("A","C"), ("B","D"), ("C","D"), ("C","E")])
    print(g1)
    print("topological sort:", g1.topological_sort())
    print("all paths A→D   :", g1.all_paths("A", "D"))
    print("longest  A→D    :", g1.longest_path("A", "D"))
    print("spanning tree   :", g1.spanning_tree())
    dom1 = g1.minimum_dominating_sets()
    print("min dominating  :", dom1)
    print("  formatted (:) :", format_solutions(dom1, sep=":"))

    print()
    print(SEP)
    print("EXAMPLE 2 — Build dependency graph")
    print(SEP)
    g2 = DAG.from_adjacency_dict({
        "app":   ["ui", "db"],
        "ui":    ["icons", "core"],
        "db":    ["core"],
        "icons": [],
        "core":  [],
    })
    print(g2)
    print("build order:", g2.topological_sort())
    print("all paths app→core:", g2.all_paths("app", "core"))

    print()
    print(SEP)
    print("CYCLE DETECTION — expected ValueError")
    print(SEP)
    try:
        DAG(edges=[("A","B"), ("B","C"), ("C","A")])
    except ValueError as e:
        print(f"Caught: {e}")

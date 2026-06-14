"""
graphs.py — Five graph types + classic algorithms.
No external imports; only plain Python built-ins.

========================================================================
WHAT IS A GRAPH?
A graph is a collection of NODES (things) connected by EDGES (links).
  city map  → nodes = cities,   edges = roads
  internet  → nodes = websites, edges = hyperlinks
  schedule  → nodes = tasks,    edges = "must-do-first" arrows

========================================================================
THE FIVE TYPES (from most general to most specific)
========================================================================

 GRAPH  —  undirected, may have cycles
 ─────────────────────────────────────────────────────────────────────
   Edges have no direction: A-B is the same as B-A.
   Can loop back: A-B-C-A is allowed.
   Think: roads, friendships, power grids.

   nodes()                    list of all node names
   edges()                    list of all connections as (A,B) pairs
   neighbors(node)            who is directly connected to node?
   degree(node)               how many edges does node have?
   has_path(src,dst)          can you reach dst starting from src?
   is_connected()             can every node reach every other node?
   connected_components()     which groups of nodes form separate islands?
   all_paths(src,dst)         every possible route from src to dst
   minimum_dominating_sets()  smallest set S so every node outside S
                              is exactly one step from a member of S


 ACYCLICGRAPH  —  undirected, no cycles  (inherits all Graph methods)
 ─────────────────────────────────────────────────────────────────────
   Like Graph but guaranteed to contain no loops.
   A connected AcyclicGraph IS a tree; disconnected = a forest.
   Raises an error if you try to add a cycle.


 DIRECTEDGRAPH  —  directed, may have cycles
 ─────────────────────────────────────────────────────────────────────
   Edges have a direction: A→B does NOT mean B→A.
   Can loop back: A→B→C→A is allowed.
   Think: one-way streets, Twitter follows, web links.

   nodes(), edges()           same idea as Graph
   children(node)             nodes this node points TO
   parents(node)              nodes that point TO this node
   in_degree(node)            how many arrows arrive at node?
   out_degree(node)           how many arrows leave node?
   has_path(src,dst)          can you reach dst following arrows?
   has_cycle()                does any directed loop exist?
   transpose()                return a new graph with every arrow flipped
   strongly_connected_components()
                              groups of nodes where every node can reach
                              every other (Kosaraju, two DFS passes)


 DAG  —  directed, no cycles  (inherits all DirectedGraph methods)
 ─────────────────────────────────────────────────────────────────────
   Directed AND guaranteed to have no directed cycles.
   Raises an error if you try to add a cycle.
   Think: dependency graphs, build systems, DP subproblems, schedules.

   roots()                    nodes with no incoming arrows (entry points)
   leaves()                   nodes with no outgoing arrows (dead ends)
   topological_sort()         order nodes so each one comes after everything
                              it depends on  (Kahn's algorithm, BFS)
   descendants(node)          all nodes reachable from node following arrows
   ancestors(node)            all nodes that can reach node
   all_paths(src,dst)         every directed route from src to dst
   shortest_path(src,dst)     fewest-step route  (BFS)
   longest_path(src,dst)      most-step route    (DP — easy on a DAG!)


 TREE  —  rooted directed tree  (inherits all DAG methods)
 ─────────────────────────────────────────────────────────────────────
   A rooted DAG where every non-root node has EXACTLY ONE parent.
   Raises an error if any non-root node has more or fewer than one parent.
   Think: file systems, XML, JSON, recursive call stacks.

   parent(node)               the one node directly above this one
   depth(node)                how many steps down from the root?
   height()                   how deep is the deepest leaf?
   is_leaf(node)              does this node have no children?
   nodes_at_depth(d)          all nodes exactly d steps from the root
   subtree(node)              cut out the piece of tree below node
   siblings(node)             other children of the same parent
   lca(a,b)                   Lowest Common Ancestor — deepest node that
                              is an ancestor of BOTH a and b
   pre_order()                visit order: parent BEFORE children
   post_order()               visit order: children BEFORE parent

========================================================================
STANDALONE ALGORITHMS
========================================================================
   format_solutions(solutions,sep)  readable output for solution lists
   tsp(graph,distances,start)       Traveling Salesman: shortest round trip
                                    visiting all cities  (backtracking)
   build_hanoi_tree(n)              call tree for Tower of Hanoi(n)

========================================================================
"""


# ====================================================================== #
#  Utility                                                                #
# ====================================================================== #

def format_solutions(solutions, sep=":"):
    """
    Convert a list of solutions into readable strings.

    Each solution is a tuple/list of node names joined by `sep`.

    >>> format_solutions([("A","B","C"), ("A","C")], sep=",")
    ['A,B,C', 'A,C']
    """
    return [sep.join(str(n) for n in sol) for sol in solutions]


# ====================================================================== #
#  Graph  —  undirected, may have cycles                                  #
# ====================================================================== #

class Graph:
    """
    Undirected graph.  Edges have no direction; A-B and B-A are the same.
    May contain cycles (A-B-C-A is allowed).

    _adj : dict[str, list[str]]   symmetric adjacency list
    """

    def __init__(self, edges=None, nodes=None):
        """
        edges : list of (a, b) tuples  — unordered pairs
        nodes : optional list of str   — to include isolated nodes
        """
        self._adj = {}
        if nodes:
            for n in nodes:
                self._ensure(n)
        if edges:
            for a, b in edges:
                self._ensure(a)
                self._ensure(b)
                if b not in self._adj[a]:
                    self._adj[a].append(b)
                if a not in self._adj[b]:
                    self._adj[b].append(a)

    def _ensure(self, n):
        if n not in self._adj:
            self._adj[n] = []

    # ── simple accessors ──────────────────────────────────────────────

    def nodes(self):
        """Sorted list of all node names."""
        return sorted(self._adj)

    def edges(self):
        """List of edges as sorted (a, b) tuples — each pair appears once."""
        seen, result = set(), []
        for a in sorted(self._adj):
            for b in self._adj[a]:
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    def neighbors(self, node):
        """All nodes directly connected to node."""
        return list(self._adj.get(node, []))

    def degree(self, node):
        """Number of edges incident to node."""
        return len(self._adj.get(node, []))

    def __str__(self):
        lines = [f"{self.__class__.__name__}:"]
        for n in self.nodes():
            lines.append(f"  {n}  neighbors={sorted(self._adj[n])}")
        return "\n".join(lines)

    # ── core algorithms ────────────────────────────────────────────────

    def has_path(self, src, dst):
        """
        ITERATIVE — BFS.
        True if any path from src to dst exists (following edges freely).
        Short-circuits as soon as dst is found.
        """
        if src == dst:
            return True
        visited = {src}
        queue   = [src]
        while queue:
            node = queue.pop(0)
            for v in self._adj.get(node, []):
                if v == dst:
                    return True
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return False

    def is_connected(self):
        """
        RECURSIVE — DFS.
        True when all nodes can be reached from a single starting node
        (i.e. the graph is one single component, no isolated islands).
        """
        if not self._adj:
            return True
        visited = set()

        def dfs(u):
            visited.add(u)
            for v in self._adj[u]:
                if v not in visited:
                    dfs(v)

        dfs(next(iter(self._adj)))
        return len(visited) == len(self._adj)

    def connected_components(self):
        """
        RECURSIVE — DFS over all unvisited starting nodes.
        Returns a list of components; each component is a sorted list
        of node names that can reach each other.

        A graph is connected when this returns a list with exactly one element.
        """
        visited    = set()
        components = []

        def dfs(u, comp):
            visited.add(u)
            comp.append(u)
            for v in self._adj[u]:
                if v not in visited:
                    dfs(v, comp)

        for start in self.nodes():
            if start not in visited:
                comp = []
                dfs(start, comp)
                components.append(sorted(comp))

        return components

    def all_paths(self, src, dst):
        """
        RECURSIVE — DFS with backtracking.
        Returns every route from src to dst that visits each node at most once.
        Warning: can be exponentially many paths on dense graphs.

        Works even when the graph has cycles — the on_path set prevents
        revisiting nodes already in the current path.
        """
        result = []

        def dfs(u, path, on_path):
            if u == dst:
                result.append(list(path))
                return
            for v in self._adj.get(u, []):
                if v not in on_path:
                    on_path.add(v)
                    path.append(v)
                    dfs(v, path, on_path)
                    path.pop()
                    on_path.remove(v)   # backtrack

        dfs(src, [src], {src})
        return result

    def minimum_dominating_sets(self):
        """
        RECURSIVE — backtracking with size pruning + feasibility pruning.

        Find all minimum dominating sets S:
          every node NOT in S must be adjacent to (= one hop from) some node in S.

        This is the undirected Minimum Set Cover — NP-complete in general.

        Pruning 1 — size:        if |chosen| >= best, abandon branch.
        Pruning 2 — feasibility: if some uncovered node has no remaining
                                  candidate that can cover it, abandon.

        Returns a list of optimal solutions (each a sorted list of nodes).
        """
        nodes     = self.nodes()
        n         = len(nodes)
        all_nodes = set(nodes)

        coverage    = {u: set(self.neighbors(u)) | {u} for u in nodes}
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
            if len(chosen) >= best_size[0]:                # pruning 1
                return
            for v in all_nodes - covered:                  # pruning 2
                if not any(i >= idx for i in coverers_of[v]):
                    return
            if idx >= n:
                return
            node = nodes[idx]
            chosen.append(node)                            # branch A: include
            backtrack(idx + 1, chosen, covered | coverage[node])
            chosen.pop()
            backtrack(idx + 1, chosen, covered)            # branch B: skip

        backtrack(0, [], set())
        return [sorted(r) for r in results]


# ====================================================================== #
#  AcyclicGraph  —  undirected, no cycles  (= forest)                    #
# ====================================================================== #

class AcyclicGraph(Graph):
    """
    Undirected graph guaranteed to contain no cycles.
    A connected AcyclicGraph is a tree; disconnected is a forest.
    Raises ValueError on construction if a cycle is detected.

    Inherits all Graph methods.
    """

    def __init__(self, edges=None, nodes=None):
        super().__init__(edges=edges, nodes=nodes)
        if self._has_cycle():
            raise ValueError("Edges form a cycle — not a valid AcyclicGraph.")

    def _has_cycle(self):
        """
        RECURSIVE — DFS tracking the parent edge.

        In an undirected graph a cycle exists when DFS reaches an already-
        visited node that is NOT the immediate parent (a back-edge).
        Returns True when a cycle is found.
        """
        visited = set()

        def dfs(u, parent):
            visited.add(u)
            for v in self._adj[u]:
                if v == parent:
                    continue            # the edge we arrived on — not a cycle
                if v in visited:
                    return True         # back-edge → cycle detected
                if dfs(v, u):
                    return True
            return False

        for start in self._adj:
            if start not in visited:
                if dfs(start, None):
                    return True
        return False


# ====================================================================== #
#  DirectedGraph  —  directed, may have cycles                            #
# ====================================================================== #

class DirectedGraph:
    """
    Directed graph.  Edge A→B does NOT mean B→A.
    May contain directed cycles (A→B→C→A is allowed).

    _forward_adj : dict[str, list[str]]   node -> list of children
    _reverse_adj : dict[str, list[str]]   node -> list of parents
    """

    def __init__(self, edges=None, nodes=None):
        """
        edges : list of (src, dst) tuples
        nodes : optional list of str (to include isolated nodes)
        """
        self._forward_adj = {}
        self._reverse_adj = {}
        if nodes:
            for n in nodes:
                self._ensure(n)
        if edges:
            for src, dst in edges:
                self._link(src, dst)

    def _ensure(self, n):
        if n not in self._forward_adj:
            self._forward_adj[n] = []
            self._reverse_adj[n] = []

    def _link(self, src, dst):
        self._ensure(src)
        self._ensure(dst)
        if dst not in self._forward_adj[src]:
            self._forward_adj[src].append(dst)
        if src not in self._reverse_adj[dst]:
            self._reverse_adj[dst].append(src)

    # ── simple accessors ──────────────────────────────────────────────

    def nodes(self):
        return sorted(self._forward_adj)

    def edges(self):
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
        """Number of incoming arrows (= number of parents)."""
        return len(self._reverse_adj.get(node, []))

    def out_degree(self, node):
        """Number of outgoing arrows (= number of children)."""
        return len(self._forward_adj.get(node, []))

    def __str__(self):
        lines = [f"{self.__class__.__name__}:"]
        for n in self.nodes():
            lines.append(
                f"  {n}  parents={sorted(self._reverse_adj[n])}"
                f"  children={sorted(self._forward_adj[n])}"
            )
        return "\n".join(lines)

    # ── core algorithms ────────────────────────────────────────────────

    def has_path(self, src, dst):
        """
        RECURSIVE — simple DFS.
        True if a directed path src→…→dst exists (following arrow directions).
        Short-circuits as soon as dst is found.
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

    def has_cycle(self):
        """
        RECURSIVE — three-colour DFS.

        WHITE=0 not visited | GRAY=1 on recursion stack | BLACK=2 done.
        Reaching a GRAY node is a back-edge → directed cycle exists.
        Returns True when a directed cycle is found.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._forward_adj}

        def dfs(u):
            color[u] = GRAY
            for v in self._forward_adj[u]:
                if color[v] == GRAY:
                    return True         # back-edge → cycle
                if color[v] == WHITE and dfs(v):
                    return True
            color[u] = BLACK
            return False

        return any(dfs(n) for n in list(self._forward_adj) if color[n] == WHITE)

    def transpose(self):
        """
        Return a new DirectedGraph with every arrow reversed.
        If original has A→B, the transpose has B→A.
        Useful as a building block for strongly_connected_components().
        """
        return DirectedGraph(
            edges=[(d, s) for s, d in self.edges()],
            nodes=self.nodes()
        )

    def strongly_connected_components(self):
        """
        RECURSIVE — Kosaraju's algorithm (two DFS passes), O(V + E).

        A Strongly Connected Component (SCC) is a maximal set of nodes
        where every node can reach every other node following arrows.

        Pass 1: DFS on the original graph — record finish order.
        Pass 2: DFS on the transposed graph in REVERSE finish order.
                Each DFS tree in pass 2 is one SCC.

        Returns a list of sorted node lists (one per SCC).
        """
        # ── Pass 1: record finish order ───────────────────────────────
        visited  = set()
        finish   = []          # nodes in order of DFS completion

        def dfs1(u):
            visited.add(u)
            for v in self._forward_adj.get(u, []):
                if v not in visited:
                    dfs1(v)
            finish.append(u)

        for n in self.nodes():
            if n not in visited:
                dfs1(n)

        # ── Pass 2: DFS on transposed graph in reverse finish order ───
        T       = self.transpose()
        visited = set()
        sccs    = []

        def dfs2(u, comp):
            visited.add(u)
            comp.append(u)
            for v in T._forward_adj.get(u, []):
                if v not in visited:
                    dfs2(v, comp)

        for node in reversed(finish):
            if node not in visited:
                comp = []
                dfs2(node, comp)
                sccs.append(sorted(comp))

        return sccs


# ====================================================================== #
#  DAG  —  Directed Acyclic Graph                                         #
# ====================================================================== #

class DAG(DirectedGraph):
    """
    Directed Acyclic Graph.
    Extends DirectedGraph by guaranteeing no directed cycles.
    Raises ValueError on construction if a cycle is detected.

    Inherits all DirectedGraph methods (has_path, has_cycle, transpose, …).
    """

    def __init__(self, edges=None, nodes=None):
        super().__init__(edges=edges, nodes=nodes)
        if self.has_cycle():
            raise ValueError("Edges form a directed cycle — not a valid DAG.")

    # ── simple accessors ──────────────────────────────────────────────

    def roots(self):
        """Nodes with no incoming arrows — the entry points."""
        return sorted(n for n in self._forward_adj if not self._reverse_adj[n])

    def leaves(self):
        """Nodes with no outgoing arrows — the dead ends / final outputs."""
        return sorted(n for n in self._forward_adj if not self._forward_adj[n])

    # ── core algorithms ────────────────────────────────────────────────

    def topological_sort(self):
        """
        ITERATIVE — Kahn's algorithm (BFS over in-degree counts).

        Returns nodes in an order where every node appears only AFTER
        all the nodes it depends on (all its predecessors).
        Alphabetical tie-breaking makes the result deterministic.
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

    def descendants(self, node):
        """
        RECURSIVE — simple DFS (forward graph).
        All nodes reachable FROM node following arrows (node itself excluded).
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
        All nodes that can reach node by following arrows (node itself excluded).
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

    def all_paths(self, src, dst):
        """
        RECURSIVE — DFS with backtracking.
        Every directed route from src to dst as a list of node lists.
        No cycle-detection needed here because a DAG can never cycle.
        Warning: can return exponentially many paths on a dense DAG.
        """
        result = []

        def dfs(u, path, on_path):
            if u == dst:
                result.append(list(path))
                return
            for v in self._forward_adj.get(u, []):
                if v not in on_path:
                    on_path.add(v)
                    path.append(v)
                    dfs(v, path, on_path)
                    path.pop()
                    on_path.remove(v)

        dfs(src, [src], {src})
        return result

    def shortest_path(self, src, dst):
        """
        ITERATIVE — BFS (list used as FIFO queue).
        Path with the fewest edges from src to dst, or [] if no path.
        BFS guarantees the first time dst is reached = shortest route.
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
        Path with the most edges from src to dst, or [] if no path.

        dist[v] = max edge-count from src to v  (-1 = not yet reached).
        prev[v] = predecessor on the best known path to v.

        We process nodes in topological order, so when we relax u's edges,
        dist[u] is already its final value (nothing can improve it later).
        This is what makes longest path tractable on a DAG but NP-hard
        on general graphs with cycles.
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


# ====================================================================== #
#  Tree  —  rooted directed tree (one parent per non-root node)           #
# ====================================================================== #

class Tree(DAG):
    """
    Rooted directed tree.
    Extends DAG by requiring that every non-root node has EXACTLY one parent.
    Raises ValueError if that constraint is violated.

    Inherits all DAG methods (topological_sort, descendants, ancestors,
    shortest_path, longest_path, all_paths, …).
    """

    def __init__(self, root, edges=None):
        """
        root  : str — the root node name
        edges : list of (parent, child) tuples
        """
        super().__init__(edges=edges, nodes=[root])
        for n in self.nodes():
            if n != root and self.in_degree(n) != 1:
                raise ValueError(
                    f"Node {n!r} has in-degree {self.in_degree(n)}"
                    f" — every non-root must have exactly one parent.")
        self.root = root

    # ── accessors ─────────────────────────────────────────────────────

    def parent(self, node):
        """The unique parent of node, or None when node is the root."""
        p = self.parents(node)
        return p[0] if p else None

    def is_leaf(self, node):
        """True when node has no children (is a leaf node)."""
        return self.out_degree(node) == 0

    def siblings(self, node):
        """Other children of the same parent (empty list for the root)."""
        par = self.parent(node)
        if par is None:
            return []
        return sorted(c for c in self.children(par) if c != node)

    # ── depth / height ─────────────────────────────────────────────────

    def depth(self, node):
        """
        RECURSIVE.
        Number of steps from the root down to node (root has depth 0).
        """
        if node == self.root:
            return 0
        return 1 + self.depth(self.parent(node))

    def height(self):
        """
        RECURSIVE — DFS from root.
        Depth of the deepest leaf (root-only tree has height 0).
        More efficient than max(depth(n) for n in …) because it
        visits each node once rather than tracing root→node per leaf.
        """
        def dfs(u):
            ch = self.children(u)
            if not ch:
                return 0
            return 1 + max(dfs(v) for v in ch)

        return dfs(self.root)

    def nodes_at_depth(self, d):
        """
        RECURSIVE — DFS.
        All nodes exactly d steps below the root, returned sorted.
        """
        result = []

        def dfs(u, current):
            if current == d:
                result.append(u)
                return
            for v in self.children(u):
                dfs(v, current + 1)

        dfs(self.root, 0)
        return sorted(result)

    # ── structural extraction ──────────────────────────────────────────

    def subtree(self, node):
        """
        RECURSIVE — DFS.
        Return a new Tree containing `node` and all its descendants,
        with `node` as the new root.
        """
        edges = []

        def dfs(u):
            for v in self.children(u):
                edges.append((u, v))
                dfs(v)

        dfs(node)
        return Tree(root=node, edges=edges)

    def lca(self, a, b):
        """
        Lowest Common Ancestor — the deepest node that is an ancestor
        of BOTH a and b (or a/b itself if one is an ancestor of the other).

        Strategy: walk from a up to root collecting all visited ancestors,
        then walk from b up until we hit a node in that set.
        O(depth(a) + depth(b)).
        """
        path_a = set()
        node   = a
        while node is not None:
            path_a.add(node)
            node = self.parent(node)

        node = b
        while node is not None:
            if node in path_a:
                return node
            node = self.parent(node)

        return None    # only happens if a and b are in different trees

    # ── traversals ────────────────────────────────────────────────────

    def pre_order(self):
        """
        RECURSIVE — DFS, root first (parent visited BEFORE its children).
        Matches the order in which a recursive function is called.
        """
        result = []

        def dfs(u):
            result.append(u)
            for v in self.children(u):
                dfs(v)

        dfs(self.root)
        return result

    def post_order(self):
        """
        RECURSIVE — DFS, leaves first (children visited BEFORE parent).
        Matches the order in which a recursive function returns.
        """
        result = []

        def dfs(u):
            for v in self.children(u):
                dfs(v)
            result.append(u)

        dfs(self.root)
        return result


# ====================================================================== #
#  Standalone algorithms                                                  #
# ====================================================================== #

def tsp(graph, distances, start=None):
    """
    RECURSIVE — DFS with backtracking.

    Find a minimum-cost Hamiltonian cycle: visit every node exactly once
    and return to the starting node using the least total distance.

    graph     : Graph  (typically a complete undirected graph)
    distances : dict of {(a, b): cost}  — symmetric (both orders looked up)
    start     : starting node; defaults to the first alphabetical node

    Returns (min_cost, path) where path includes the return to start.
    Complexity: O(n!) — practical only for small n (~10 or fewer cities).
    """
    nodes = graph.nodes()
    if not nodes:
        return 0, []
    if start is None:
        start = nodes[0]

    def dist(a, b):
        return distances.get((a, b)) or distances.get((b, a), float("inf"))

    best = [float("inf"), None]

    def backtrack(current, visited, cost, path):
        if len(visited) == len(nodes):
            total = cost + dist(current, start)
            if total < best[0]:
                best[0] = total
                best[1] = path + [start]
            return
        for n in sorted(graph.neighbors(current)):
            if n not in visited:
                visited.add(n)
                backtrack(n, visited, cost + dist(current, n), path + [n])
                visited.remove(n)

    backtrack(start, {start}, 0, [start])
    return best[0], best[1]


def build_hanoi_tree(n, src="A", dst="C", via="B"):
    """
    RECURSIVE — build the call tree for Tower of Hanoi(n discs).

    Each tree node = one recursive call, labelled 'h(discs,src>dst)#id'.
    Unique '#id' suffix avoids duplicate labels when the same call
    (n, src, dst) appears in multiple branches.

    Pre-order  traversal = order calls are MADE (recursive descent).
    Post-order traversal = order calls RETURN (includes base-case moves).
    Leaves               = base cases (n=1): one concrete disc move each.

    Returns a Tree.
    """
    uid   = [0]
    root  = [None]
    edges = []

    def rec(n, src, dst, via, parent_label):
        uid[0] += 1
        label = f"h({n},{src}>{dst})#{uid[0]}"
        if parent_label is None:
            root[0] = label
        else:
            edges.append((parent_label, label))
        if n > 1:
            rec(n - 1, src, via, dst, label)   # move top (n-1) discs: src → via
            rec(n - 1, via, dst, src, label)   # move top (n-1) discs: via → dst

    rec(n, src, dst, via, None)
    return Tree(root=root[0], edges=edges)


# ====================================================================== #
#  __main__  —  five example problems                                     #
# ====================================================================== #

if __name__ == "__main__":

    SEP = "=" * 62

    # ------------------------------------------------------------------ #
    #  1. Fibonacci — DAG of overlapping subproblems                     #
    # ------------------------------------------------------------------ #
    # Edge f(a) → f(b) means "computing f(a) needs f(b) first".
    # Topological sort = memoized top-down call order.
    # Nodes with in_degree > 1 are shared subproblems → saved by memoization.

    print(SEP)
    print("1. FIBONACCI — subproblem DAG")
    print(SEP)

    fib = DAG(edges=[
        ("f5", "f4"), ("f5", "f3"),
        ("f4", "f3"), ("f4", "f2"),
        ("f3", "f2"), ("f3", "f1"),
        ("f2", "f1"), ("f2", "f0"),
    ])

    order  = fib.topological_sort()
    shared = [n for n in fib.nodes() if fib.in_degree(n) > 1]

    print("Evaluation order (topological sort):")
    print(" ", " → ".join(order))
    print("Shared subproblems (in-degree > 1):", shared)
    print("  → without memoization these are recomputed many times")
    print("has_path f5→f0 :", fib.has_path("f5", "f0"))
    print("all paths f5→f0:", fib.all_paths("f5", "f0"))

    # ------------------------------------------------------------------ #
    #  2. Tower of Hanoi — recursive call Tree                           #
    # ------------------------------------------------------------------ #
    # Pre-order  = order calls are MADE (push onto call stack).
    # Post-order = order calls RETURN (each leaf = one disc move).

    print()
    print(SEP)
    print("2. TOWER OF HANOI — recursive call Tree (n=3)")
    print(SEP)

    ht   = build_hanoi_tree(n=3)
    pre  = ht.pre_order()
    post = ht.post_order()

    print(f"Nodes: {len(ht.nodes())} (= 2^n-1)   height: {ht.height()}")
    print(f"LCA of first and last leaf: {ht.lca(pre[-1], pre[1])}")
    print()
    print("Pre-order (call sequence):")
    for node in pre:
        d = ht.depth(node)
        print(f"  {'  '*d}→ {node}")
    print()
    print("Leaf nodes = actual disc moves (post-order):")
    for node in post:
        if ht.is_leaf(node):
            print(f"  {node}")

    # ------------------------------------------------------------------ #
    #  3. Task Scheduling — DAG topological sort + critical path         #
    # ------------------------------------------------------------------ #

    print()
    print(SEP)
    print("3. TASK SCHEDULING — DAG")
    print(SEP)

    tasks = DAG(edges=[
        ("design",   "backend"),
        ("design",   "frontend"),
        ("backend",  "tests"),
        ("frontend", "tests"),
        ("tests",    "deploy"),
    ])

    print("Valid schedule  :", " → ".join(tasks.topological_sort()))
    print("Critical path   :", " → ".join(tasks.longest_path("design", "deploy")))
    print("Shortest path   :", " → ".join(tasks.shortest_path("design", "deploy")))
    print("All paths       :", tasks.all_paths("design", "deploy"))
    print("Descendants of design:", tasks.descendants("design"))

    # ------------------------------------------------------------------ #
    #  4. Sensor Network Coverage — Graph minimum dominating set         #
    # ------------------------------------------------------------------ #
    # Edges are bidirectional (roads), so we use Graph, not DAG.
    # Note: this graph has TWO disconnected components.

    print()
    print(SEP)
    print("4. SENSOR NETWORK COVERAGE — Graph")
    print(SEP)

    locaties = Graph(edges=[
        ("R", "Q"), ("U", "Z"), ("K", "Y"), ("Q", "Z"), ("L", "K")
    ])

    comps     = locaties.connected_components()
    solutions = locaties.minimum_dominating_sets()

    print(f"Nodes      : {locaties.nodes()}")
    print(f"Connected? : {locaties.is_connected()}  → two separate islands!")
    print(f"Components : {comps}")
    print(f"  → minimum relay set needs at least one node from each island")
    print()
    print(f"Minimum relay size : {len(solutions[0])}")
    print("All optimal relay sets:")
    for s in format_solutions(solutions, sep=", "):
        print(f"  {{ {s} }}")
    print()
    print("has_path K→Z :", locaties.has_path("K", "Z"))
    print("all_paths R→Z:", locaties.all_paths("R", "Z"))

    # ------------------------------------------------------------------ #
    #  5. Traveling Salesman — backtracking on complete Graph            #
    # ------------------------------------------------------------------ #

    print()
    print(SEP)
    print("5. TRAVELING SALESMAN — backtracking on complete Graph")
    print(SEP)

    cities = Graph(edges=[
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "C"), ("B", "D"), ("C", "D"),
    ])

    d = {
        ("A", "B"): 2, ("A", "C"): 9, ("A", "D"): 10,
        ("B", "C"): 6, ("B", "D"): 4, ("C", "D"):  3,
    }

    cost, path = tsp(cities, d, start="A")

    print(f"Cities     : {cities.nodes()}")
    print(f"Connected? : {cities.is_connected()}")
    print(f"Optimal tour (from A) : {' → '.join(path)}")
    print(f"Total distance        : {cost}")
    print("  → A→B(2)+B→D(4)+D→C(3)+C→A(9) = 18")

    # ------------------------------------------------------------------ #
    #  Cycle / SCC demo                                                   #
    # ------------------------------------------------------------------ #

    print()
    print(SEP)
    print("CYCLE DETECTION + SCC demo")
    print(SEP)

    dg = DirectedGraph(edges=[
        ("A", "B"), ("B", "C"), ("C", "A"),   # SCC: {A,B,C}
        ("B", "D"),                            # D only reachable, not part of cycle
    ])
    print("DirectedGraph A→B→C→A (+ B→D)")
    print("  has_cycle():", dg.has_cycle())
    print("  strongly_connected_components():", dg.strongly_connected_components())
    print("  transpose edges:", dg.transpose().edges())

    try:
        DAG(edges=[("X", "Y"), ("Y", "Z"), ("Z", "X")])
    except ValueError as e:
        print(f"\nDAG(X→Y→Z→X): {e}")

    try:
        AcyclicGraph(edges=[("P", "Q"), ("Q", "R"), ("R", "P")])
    except ValueError as e:
        print(f"AcyclicGraph(P-Q-R-P): {e}")

    # ------------------------------------------------------------------ #
    #  Tree extras demo                                                   #
    # ------------------------------------------------------------------ #

    print()
    print(SEP)
    print("TREE extras — subtree, lca, nodes_at_depth, siblings")
    print(SEP)

    #        root
    #       /    \
    #      A      B
    #     / \      \
    #    C   D      E

    t = Tree(root="root", edges=[
        ("root", "A"), ("root", "B"),
        ("A", "C"),    ("A", "D"),
        ("B", "E"),
    ])

    print("height()         :", t.height())
    print("nodes_at_depth(2):", t.nodes_at_depth(2))
    print("lca(C, E)        :", t.lca("C", "E"))
    print("lca(C, D)        :", t.lca("C", "D"))
    print("siblings(C)      :", t.siblings("C"))
    print("subtree(A).nodes():", t.subtree("A").nodes())
    print("is_leaf(C)       :", t.is_leaf("C"))
    print("is_leaf(A)       :", t.is_leaf("A"))

"""
graphs.py — Educational graph class hierarchy.
No external imports; only plain Python built-ins.

Pass  debug=True  to any constructor to get step-by-step explanations
of what the internal data structures look like and how each algorithm
makes its decisions.  Great for learning; turn it off once you understand.

Class hierarchy
---------------
  Graph                  undirected, may have cycles
  AcyclicGraph(Graph)    undirected, no cycles  (= forest)
  DirectedGraph          directed,   may have cycles
  DAG(DirectedGraph)     directed,   no cycles
  Tree(DAG)              rooted directed tree

Standalone algorithms
---------------------
  format_solutions(solutions, sep)   pretty-print solution lists
  tsp(graph, distances, start)       Traveling Salesman — backtracking
  build_hanoi_tree(n)                Tower of Hanoi recursive call tree
"""


# ====================================================================== #
#  Utility                                                                #
# ====================================================================== #

def format_solutions(solutions, sep=":"):
    """
    Convert a list of solution-tuples into a list of joined strings.

    Parameters
    ----------
    solutions : list of tuple or list   each element is one solution
    sep       : str                     separator between node names

    Returns
    -------
    list of str

    Example
    -------
    >>> format_solutions([('A','B','C'), ('A','C')], sep=':')
    ['A:B:C', 'A:C']
    >>> format_solutions([('K','Q','U')], sep=', ')
    ['K, Q, U']
    """
    return [sep.join(str(n) for n in sol) for sol in solutions]


# ====================================================================== #
#  Graph  —  undirected, may have cycles                                  #
# ====================================================================== #

class Graph:
    """
    Undirected graph.  Edges have no direction; A-B and B-A are the same.
    May contain cycles.

    Internal storage
    ----------------
    _adj : dict[str, list[str]]   symmetric adjacency list.

    debug parameter
    ---------------
    Pass debug=True to print step-by-step explanations of internal data
    structures and algorithm decisions.  Useful for learning.

    Example
    -------
    >>> g = Graph(edges=[('A','B'), ('B','C'), ('A','C')])
    >>> g = Graph(nodes=['X','Y','Z'], edges=[('X','Y')])  # Z is isolated
    >>> g = Graph(debug=True)                              # empty, with debug on
    """

    def __init__(self, edges=None, nodes=None, debug=False):
        """
        Parameters
        ----------
        edges : list of (str, str) tuples   unordered pairs, e.g. [('A','B'), ('B','C')]
        nodes : list of str (optional)      explicit node names including isolated nodes
        debug : bool (default False)        print educational step-by-step output

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('A','C'), ('B','D')])
        >>> g = Graph(nodes=['P','Q','R'], edges=[('P','Q')])  # R has no edges
        """
        self._debug = debug
        self._adj   = {}

        if nodes:
            for n in nodes:
                # register node n if it does not yet exist
                if n not in self._adj:
                    self._adj[n] = []
        if edges:
            for a, b in edges:
                # register each endpoint if it does not yet exist
                if a not in self._adj:
                    self._adj[a] = []
                if b not in self._adj:
                    self._adj[b] = []
                if b not in self._adj[a]:
                    self._adj[a].append(b)
                if a not in self._adj[b]:
                    self._adj[b].append(a)

        if self._debug:
            print(f"\n{'='*60}")
            print(f"[{self.__class__.__name__}.__init__] Internal data structure")
            print(f"{'='*60}")
            print("  _adj  =  adjacency list")
            print("  Maps every node to the list of its direct neighbours.")
            print("  Because the graph is UNDIRECTED, every edge A-B is stored")
            print("  in BOTH directions: adj[A] contains B  AND  adj[B] contains A.")
            print()
            if self._adj:
                for node in sorted(self._adj):
                    nbrs = sorted(self._adj[node])
                    print(f"  _adj[{node!r}] = {nbrs}")
            else:
                print("  (no nodes yet)")
            print()

    # ── simple accessors ──────────────────────────────────────────────

    def nodes(self):
        """
        Sorted list of all node names (strings).

        Example
        -------
        >>> g = Graph(edges=[('B','A'), ('C','A')])
        >>> g.nodes()
        ['A', 'B', 'C']
        """
        return sorted(self._adj)

    def edges(self):
        """
        List of edges as sorted (a, b) tuples — each undirected pair appears once.

        Example
        -------
        >>> g = Graph(edges=[('B','A'), ('C','B')])
        >>> g.edges()
        [('A', 'B'), ('B', 'C')]
        """
        seen, result = set(), []
        for a in sorted(self._adj):
            for b in self._adj[a]:
                key = tuple(sorted([a, b]))
                if key not in seen:
                    seen.add(key)
                    result.append(key)
        return result

    def neighbors(self, node):
        """
        List of nodes directly connected to node (unsorted).

        Parameters
        ----------
        node : str   a node name that exists in the graph

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('A','C'), ('B','D')])
        >>> g.neighbors('A')
        ['B', 'C']
        >>> g.neighbors('D')
        ['B']
        """
        return list(self._adj.get(node, []))

    def degree(self, node):
        """
        Number of edges incident to node (= number of neighbours).

        Parameters
        ----------
        node : str

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('A','C'), ('A','D')])
        >>> g.degree('A')
        3
        >>> g.degree('B')
        1
        """
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
        Return True if any path from src to dst exists (edges in any direction).

        Parameters
        ----------
        src : str   starting node
        dst : str   destination node

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('B','C'), ('D','E')])
        >>> g.has_path('A', 'C')
        True
        >>> g.has_path('A', 'D')   # different component
        False
        >>> g.has_path('A', 'A')   # same node -> always True
        True
        """
        if self._debug:
            print(f"\n[{self.__class__.__name__}.has_path]  '{src}' -> '{dst}'?")
            print("  Strategy: BFS — spread outward one hop at a time.")
            print("  First time we reach dst, we know a path exists.")
            print(f"  Start: queue=['{src}']  visited={{'{src}'}}")

        if src == dst:
            if self._debug:
                print("  src == dst -> True (trivially)")
            return True

        visited = {src}
        queue   = [src]
        step    = 0

        while queue:
            node = queue.pop(0)
            step += 1
            if self._debug:
                nbrs = sorted(self._adj.get(node, []))
                print(f"  Step {step}: process '{node}' -> neighbors {nbrs}")
            for v in self._adj.get(node, []):
                if v == dst:
                    if self._debug:
                        print(f"    '{v}' == dst -> path FOUND  (True)")
                    return True
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    if self._debug:
                        print(f"    enqueue '{v}'  visited={sorted(visited)}")

        if self._debug:
            print(f"  Queue empty, '{dst}' never reached -> False")
        return False

    def has_path_with_maximum_length(self, src, dst, max_length):
        """
        ITERATIVE — BFS with hop counter.
        Return True if a path from src to dst exists using at most
        max_length edges.

        Parameters
        ----------
        src        : str   starting node
        dst        : str   destination node
        max_length : int   maximum number of edges allowed on the path

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('B','C'), ('A','C')])
        >>> g.has_path_with_maximum_length('A', 'C', 1)   # direct edge A-C exists
        True
        >>> g.has_path_with_maximum_length('A', 'C', 0)   # need at least 1 hop
        False
        >>> g = Graph(edges=[('A','B'), ('B','C'), ('C','D')])
        >>> g.has_path_with_maximum_length('A', 'D', 2)   # shortest path is 3 hops
        False
        >>> g.has_path_with_maximum_length('A', 'D', 3)
        True
        """
        if src == dst:
            return True

        # BFS — each queue entry is (node, distance_so_far).
        # BFS explores in order of increasing distance, so the first time
        # we reach dst it is via the shortest path.  If that path length
        # exceeds max_length we can stop immediately.
        visited = {src}
        queue   = [(src, 0)]

        while queue:
            node, dist = queue.pop(0)
            if dist >= max_length:
                continue
            for v in self._adj.get(node, []):
                if v == dst:
                    return True
                if v not in visited:
                    visited.add(v)
                    queue.append((v, dist + 1))

        return False

    def is_connected(self):
        """
        RECURSIVE — DFS.
        Return True when all nodes can be reached from a single starting node
        (the whole graph is one piece with no isolated islands).

        Example
        -------
        >>> Graph(edges=[('A','B'), ('B','C')]).is_connected()
        True
        >>> Graph(edges=[('A','B'), ('C','D')]).is_connected()   # two islands
        False
        """
        if self._debug:
            print(f"\n[{self.__class__.__name__}.is_connected]")
            print("  Strategy: DFS from any one node.")
            print("  If all nodes are visited -> the graph is one piece (connected).")

        if not self._adj:
            return True

        start   = next(iter(self._adj))
        visited = set()

        def dfs(u):
            visited.add(u)
            if self._debug:
                print(f"    visit '{u}'")
            for v in self._adj[u]:
                if v not in visited:
                    dfs(v)

        if self._debug:
            print(f"  Start DFS from '{start}':")
        dfs(start)

        result = len(visited) == len(self._adj)
        if self._debug:
            missed = sorted(set(self._adj) - visited)
            if result:
                print(f"  All {len(visited)} nodes reached -> connected: True")
            else:
                print(f"  Only {len(visited)}/{len(self._adj)} nodes reached.")
                print(f"  Unreachable: {missed} -> connected: False")
        return result

    def connected_components(self):
        """
        RECURSIVE — DFS over all unvisited nodes.
        Return a list of components; each component is a sorted list of node names
        that can all reach each other.

        A graph is connected when this returns a list containing exactly one element.

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('C','D'), ('D','E')])
        >>> g.connected_components()
        [['A', 'B'], ['C', 'D', 'E']]
        """
        if self._debug:
            print(f"\n[{self.__class__.__name__}.connected_components]")
            print("  Strategy: repeat DFS from each unvisited node.")
            print("  Each DFS discovers one complete island (component).")

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
                if self._debug:
                    print(f"  Component #{len(components)}: {sorted(comp)}")

        if self._debug:
            print(f"  Total: {len(components)} component(s)")
        return components

    def all_paths(self, src, dst):
        """
        RECURSIVE — DFS with backtracking.
        Return every simple route from src to dst as a list of node lists.
        Each node is visited at most once per path.
        Works even when the graph has cycles (path-set prevents revisiting).
        Warning: exponentially many paths on dense graphs.

        Parameters
        ----------
        src : str   starting node
        dst : str   destination node

        Returns
        -------
        list of list[str]   e.g.  [['A','B','D'], ['A','C','D']]

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D')])
        >>> g.all_paths('A', 'D')
        [['A', 'B', 'D'], ['A', 'C', 'D']]
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

        if self._debug:
            print(f"\n[{self.__class__.__name__}.all_paths]  '{src}' -> '{dst}'")
            print(f"  Found {len(result)} path(s) (DFS + backtracking):")
            for p in result:
                print(f"    {' -> '.join(p)}")

        return result

    def minimum_dominating_sets(self):
        """
        RECURSIVE — backtracking with size pruning + feasibility pruning.
        NP-complete in general; practical for graphs up to ~20-25 nodes.

        Find all minimum dominating sets S:
          every node NOT in S must be adjacent to at least one node IN S.

        Undirected coverage: coverage(u) = {u} ∪ neighbours(u)

        Returns
        -------
        list of list[str]   all optimal solutions, each sorted alphabetically.
                            e.g.  [['A', 'C'], ['B']]

        Example
        -------
        >>> g = Graph(edges=[('A','B'), ('B','C')])
        >>> g.minimum_dominating_sets()   # B alone covers everything
        [['B']]
        >>> Graph(edges=[('A','B'),('C','D')]).minimum_dominating_sets()
        [['A', 'C'], ['A', 'D'], ['B', 'C'], ['B', 'D']]
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

        if self._debug:
            print(f"\n[{self.__class__.__name__}.minimum_dominating_sets]")
            print("  coverage(u) = {u} + neighbours(u)")
            print("  When u joins S it 'covers' itself and all its neighbours.")
            print()
            print(f"  {'Node':<8}  Coverage set")
            for u in nodes:
                print(f"  {u:<8}  {sorted(coverage[u])}")
            print()
            print("  Backtracking: for each node (sorted), two branches:")
            print("    Branch A: INCLUDE in S  (expands covered set)")
            print("    Branch B: SKIP")
            print("  Two pruning rules cut dead branches:")
            print("    Pruning 1: |chosen| >= best size found   -> abandon")
            print("    Pruning 2: some uncovered node has no remaining candidate -> abandon")
            print()

        best_size   = [n]
        results     = []
        prune1_hits = [0]
        prune2_hits = [0]

        def backtrack(idx, chosen, covered):
            if covered == all_nodes:
                size = len(chosen)
                if size < best_size[0]:
                    best_size[0] = size
                    results.clear()
                    results.append(tuple(chosen))
                    if self._debug:
                        print(f"  *** New best: {sorted(chosen)} (size {size})")
                elif size == best_size[0]:
                    results.append(tuple(chosen))
                    if self._debug:
                        print(f"  *** Tied best: {sorted(chosen)} (size {size})")
                return
            if len(chosen) >= best_size[0]:
                prune1_hits[0] += 1
                return
            for v in all_nodes - covered:
                if not any(i >= idx for i in coverers_of[v]):
                    prune2_hits[0] += 1
                    return
            if idx >= n:
                return
            node = nodes[idx]
            chosen.append(node)
            backtrack(idx + 1, chosen, covered | coverage[node])
            chosen.pop()
            backtrack(idx + 1, chosen, covered)

        backtrack(0, [], set())

        if self._debug:
            print(f"  Pruning 1 fired {prune1_hits[0]}x  (size bound)")
            print(f"  Pruning 2 fired {prune2_hits[0]}x  (feasibility)")
            print(f"  Optimal size: {best_size[0]},  {len(results)} solution(s)")

        return [sorted(r) for r in results]


# ====================================================================== #
#  AcyclicGraph  —  undirected, no cycles  (= forest)                    #
# ====================================================================== #

class AcyclicGraph(Graph):
    """
    Undirected graph guaranteed to contain no cycles (a forest).
    A connected AcyclicGraph IS a tree; disconnected = a forest.
    Raises ValueError on construction if a cycle is detected.
    Inherits all Graph methods.

    Example
    -------
    >>> g = AcyclicGraph(edges=[('A','B'), ('B','C'), ('A','D')])  # valid tree
    >>> AcyclicGraph(edges=[('A','B'), ('B','C'), ('C','A')])      # raises ValueError
    """

    def __init__(self, edges=None, nodes=None, debug=False):
        """
        Parameters
        ----------
        edges : list of (str, str) tuples   undirected edge pairs
        nodes : list of str (optional)
        debug : bool (default False)

        Example
        -------
        >>> g = AcyclicGraph(edges=[('root','A'), ('root','B'), ('A','C')])
        """
        super().__init__(edges=edges, nodes=nodes, debug=debug)
        if self._has_cycle():
            raise ValueError("Edges form a cycle — not a valid AcyclicGraph.")

    def _has_cycle(self):
        """
        RECURSIVE — DFS tracking the parent edge.

        In an undirected graph: reaching a visited node that is NOT the
        immediate parent we came from = back-edge = cycle.
        Returns True when a cycle is found.
        """
        if self._debug:
            print(f"\n[AcyclicGraph._has_cycle]  Checking for cycles")
            print("  A cycle exists when DFS reaches an already-visited node")
            print("  that is NOT the edge we just came from.")

        visited = set()

        def dfs(u, parent, depth=0):
            visited.add(u)
            indent = "  " * (depth + 1)
            if self._debug:
                print(f"{indent}visit '{u}' (came from '{parent}')")
            for v in self._adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    if self._debug:
                        print(f"{indent}  '{v}' already visited and not parent -> CYCLE!")
                    return True
                if dfs(v, u, depth + 1):
                    return True
            return False

        for start in self._adj:
            if start not in visited:
                if dfs(start, None):
                    return True

        if self._debug:
            print("  No cycle found -> valid AcyclicGraph")
        return False


# ====================================================================== #
#  DirectedGraph  —  directed, may have cycles                            #
# ====================================================================== #

class DirectedGraph:
    """
    Directed graph.  Edge A->B does NOT mean B->A.
    May contain directed cycles (A->B->C->A is allowed).

    Internal storage
    ----------------
    _forward_adj : dict[str, list[str]]   node -> list of children
    _reverse_adj : dict[str, list[str]]   node -> list of parents

    debug parameter
    ---------------
    Pass debug=True to print internal data structures and algorithm steps.

    Example
    -------
    >>> g = DirectedGraph(edges=[('A','B'), ('A','C'), ('B','D')])
    >>> g = DirectedGraph(nodes=['X','Y'], edges=[('X','Y')])
    """

    def __init__(self, edges=None, nodes=None, debug=False):
        """
        Parameters
        ----------
        edges : list of (str, str) tuples   directed pairs (src, dst)
                                            e.g. [('A','B'), ('B','C')]
        nodes : list of str (optional)      to include isolated nodes
        debug : bool (default False)

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'), ('B','C'), ('C','A')])  # cycle ok
        >>> g = DirectedGraph(debug=True)                               # with debug
        """
        self._debug        = debug
        self._forward_adj  = {}
        self._reverse_adj  = {}

        if nodes:
            for n in nodes:
                # register node n if it does not yet exist
                if n not in self._forward_adj:
                    self._forward_adj[n] = []
                    self._reverse_adj[n] = []
        if edges:
            for src, dst in edges:
                # register each endpoint if it does not yet exist
                if src not in self._forward_adj:
                    self._forward_adj[src] = []
                    self._reverse_adj[src] = []
                if dst not in self._forward_adj:
                    self._forward_adj[dst] = []
                    self._reverse_adj[dst] = []
                # add the directed edge to both adjacency dicts
                if dst not in self._forward_adj[src]:
                    self._forward_adj[src].append(dst)
                if src not in self._reverse_adj[dst]:
                    self._reverse_adj[dst].append(src)

        if self._debug:
            print(f"\n{'='*60}")
            print(f"[{self.__class__.__name__}.__init__] Internal data structures")
            print(f"{'='*60}")
            print("  _forward_adj  maps each node to its CHILDREN")
            print("                (nodes this node has an arrow pointing TO)")
            print("  _reverse_adj  maps each node to its PARENTS")
            print("                (nodes that have an arrow pointing TO this node)")
            print("  Both dicts are kept in sync so parent lookups stay cheap.")
            print()
            hdr = f"  {'Node':<10}  {'children (_forward_adj)':<30}  parents (_reverse_adj)"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for node in sorted(self._forward_adj):
                ch = sorted(self._forward_adj[node])
                pa = sorted(self._reverse_adj[node])
                print(f"  {node:<10}  {str(ch):<30}  {pa}")
            print()

    # ── simple accessors ──────────────────────────────────────────────

    def nodes(self):
        """
        Sorted list of all node names.

        Example
        -------
        >>> g = DirectedGraph(edges=[('B','A'), ('C','A')])
        >>> g.nodes()
        ['A', 'B', 'C']
        """
        return sorted(self._forward_adj)

    def edges(self):
        """
        List of directed edges as (src, dst) tuples (sorted).
        Order: A->B, A->C, B->...

        Example
        -------
        >>> g = DirectedGraph(edges=[('B','C'), ('A','B')])
        >>> g.edges()
        [('A', 'B'), ('B', 'C')]
        """
        return [(s, d)
                for s in sorted(self._forward_adj)
                for d in self._forward_adj[s]]

    def children(self, node):
        """
        Nodes that node points TO (direct successors).

        Parameters
        ----------
        node : str

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'), ('A','C'), ('B','D')])
        >>> g.children('A')
        ['B', 'C']
        >>> g.children('D')   # leaf node
        []
        """
        return list(self._forward_adj.get(node, []))

    def parents(self, node):
        """
        Nodes that point TO node (direct predecessors).

        Parameters
        ----------
        node : str

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','C'), ('B','C')])
        >>> g.parents('C')
        ['A', 'B']
        >>> g.parents('A')   # root node
        []
        """
        return list(self._reverse_adj.get(node, []))

    def in_degree(self, node):
        """
        Number of incoming arrows (= number of parents).

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','C'), ('B','C')])
        >>> g.in_degree('C')    # two arrows arrive
        2
        >>> g.in_degree('A')    # nothing points to A
        0
        """
        return len(self._reverse_adj.get(node, []))

    def out_degree(self, node):
        """
        Number of outgoing arrows (= number of children).

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'), ('A','C')])
        >>> g.out_degree('A')   # A points to B and C
        2
        >>> g.out_degree('B')
        0
        """
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
        True if a directed path src->...->dst exists (following arrow directions).

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'), ('B','C')])
        >>> g.has_path('A', 'C')   # A->B->C exists
        True
        >>> g.has_path('C', 'A')   # no arrows go backwards
        False
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

        result = dfs(src)
        if self._debug:
            print(f"\n[{self.__class__.__name__}.has_path]  '{src}' -> '{dst}': {result}")
        return result

    def has_path_with_maximum_length(self, src, dst, max_length):
        """
        RECURSIVE — DFS with backtracking and hop counter.
        Return True if a directed path src->...->dst exists using at most
        max_length edges.

        Parameters
        ----------
        src        : str   starting node
        dst        : str   destination node
        max_length : int   maximum number of directed edges allowed

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'), ('B','C'), ('A','C')])
        >>> g.has_path_with_maximum_length('A', 'C', 1)   # direct edge A->C exists
        True
        >>> g.has_path_with_maximum_length('A', 'C', 0)
        False
        >>> g = DirectedGraph(edges=[('A','B'), ('B','C'), ('C','D')])
        >>> g.has_path_with_maximum_length('A', 'D', 2)   # shortest path is 3 hops
        False
        >>> g.has_path_with_maximum_length('A', 'D', 3)
        True
        """
        if src == dst:
            return True

        # DFS with backtracking — on_path prevents revisiting nodes already
        # on the current branch; remaining counts down the edge budget.
        on_path = {src}

        def dfs(u, remaining):
            if remaining == 0:
                return False
            for v in self._forward_adj.get(u, []):
                if v == dst:
                    return True
                if v not in on_path:
                    on_path.add(v)
                    if dfs(v, remaining - 1):
                        return True
                    on_path.remove(v)       # backtrack
            return False

        result = dfs(src, max_length)
        if self._debug:
            print(f"\n[{self.__class__.__name__}.has_path_with_maximum_length]"
                  f"  '{src}' -> '{dst}' (max {max_length} hops): {result}")
        return result

    def has_cycle(self):
        """
        RECURSIVE — three-colour DFS.
        Return True if any directed cycle exists in the graph.

        WHITE = not yet visited
        GRAY  = currently on the recursion stack
        BLACK = fully explored, no cycle found below

        Example
        -------
        >>> DirectedGraph(edges=[('A','B'), ('B','C'), ('C','A')]).has_cycle()
        True
        >>> DirectedGraph(edges=[('A','B'), ('B','C')]).has_cycle()
        False
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self._forward_adj}

        if self._debug:
            print(f"\n[{self.__class__.__name__}.has_cycle]  Three-colour DFS")
            print("  WHITE  = not yet visited")
            print("  GRAY   = currently being explored (on the recursion stack RIGHT NOW)")
            print("  BLACK  = fully explored (no cycles found below this node)")
            print("  KEY: reaching a GRAY node = we looped back = DIRECTED CYCLE!")
            print()

        depth_ctr = [0]

        def dfs(u):
            indent = "  " * (depth_ctr[0] + 1)
            color[u] = GRAY
            if self._debug:
                print(f"{indent}'{u}': WHITE -> GRAY  (start exploring)")
            depth_ctr[0] += 1
            for v in self._forward_adj[u]:
                if color[v] == GRAY:
                    if self._debug:
                        print(f"{'  ' * depth_ctr[0]}  -> '{v}' is GRAY! "
                              f"Back-edge -> CYCLE DETECTED")
                    depth_ctr[0] -= 1
                    return True
                if color[v] == WHITE:
                    if self._debug:
                        print(f"{'  ' * depth_ctr[0]}  -> follow to '{v}' (WHITE)")
                    if dfs(v):
                        depth_ctr[0] -= 1
                        return True
                else:
                    if self._debug:
                        print(f"{'  ' * depth_ctr[0]}  -> '{v}' is BLACK (safe, skip)")
            color[u] = BLACK
            depth_ctr[0] -= 1
            if self._debug:
                print(f"{indent}'{u}': GRAY -> BLACK  (no cycle here)")
            return False

        for node in list(self._forward_adj):
            if color[node] == WHITE:
                if self._debug:
                    print(f"  Start DFS from '{node}' (unvisited)")
                if dfs(node):
                    return True

        if self._debug:
            print("  All nodes explored -> no cycle found")
        return False

    def transpose(self):
        """
        Return a new DirectedGraph with every arrow reversed.
        If original has A->B, the transpose has B->A.

        Returns
        -------
        DirectedGraph   a new graph; the original is not modified.

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'), ('B','C')])
        >>> g.transpose().edges()
        [('B', 'A'), ('C', 'B')]
        """
        return DirectedGraph(
            edges=[(d, s) for s, d in self.edges()],
            nodes=self.nodes(),
            debug=self._debug
        )

    def strongly_connected_components(self):
        """
        RECURSIVE — Kosaraju's algorithm (two DFS passes), O(V + E).

        A Strongly Connected Component (SCC) is a maximal set of nodes where
        EVERY node can reach EVERY other node by following arrows.
        In a DAG, every SCC contains exactly one node.

        Returns
        -------
        list of list[str]   each inner list is one SCC, sorted alphabetically.

        Example
        -------
        >>> g = DirectedGraph(edges=[('A','B'),('B','C'),('C','A'),('B','D')])
        >>> g.strongly_connected_components()
        [['A', 'B', 'C'], ['D']]
        >>> DAG(edges=[('A','B'),('B','C')]).strongly_connected_components()
        [['A'], ['B'], ['C']]    # each node is its own SCC in a DAG
        """
        if self._debug:
            print(f"\n[{self.__class__.__name__}.strongly_connected_components]")
            print("  Kosaraju's algorithm — two DFS passes:")
            print("  Pass 1: explore original graph, record the order nodes FINISH.")
            print("  Pass 2: explore TRANSPOSED graph in REVERSE finish order.")
            print("          Each DFS tree in pass 2 = one SCC.")
            print()

        visited = set()
        finish  = []

        def dfs1(u):
            visited.add(u)
            for v in self._forward_adj.get(u, []):
                if v not in visited:
                    dfs1(v)
            finish.append(u)

        for n in self.nodes():
            if n not in visited:
                dfs1(n)

        if self._debug:
            print(f"  Pass 1 finish order: {finish}")
            print()

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
                if self._debug:
                    print(f"  SCC #{len(sccs)}: {sorted(comp)}")

        return sccs


# ====================================================================== #
#  DAG  —  Directed Acyclic Graph                                         #
# ====================================================================== #

class DAG(DirectedGraph):
    """
    Directed Acyclic Graph.
    Extends DirectedGraph: guarantees no directed cycles.
    Raises ValueError on construction if a cycle is detected.

    Example
    -------
    >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D')])
    >>> g = DAG(edges=[('design','backend'), ('design','frontend'),
    ...                ('backend','tests'), ('frontend','tests')])
    """

    def __init__(self, edges=None, nodes=None, debug=False):
        """
        Parameters
        ----------
        edges : list of (str, str) tuples   directed edges (src, dst)
        nodes : list of str (optional)
        debug : bool (default False)

        Raises
        ------
        ValueError   if the supplied edges form a directed cycle

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('B','C'), ('A','C')], debug=False)
        >>> DAG(edges=[('A','B'), ('B','A')])   # raises ValueError
        """
        super().__init__(edges=edges, nodes=nodes, debug=debug)
        if self.has_cycle():
            raise ValueError("Edges form a directed cycle — not a valid DAG.")
        if self._debug:
            print("  DAG validation passed (no directed cycle).")

    # ── simple accessors ──────────────────────────────────────────────

    def roots(self):
        """
        Nodes with no incoming arrows (entry points / sources).
        In a dependency graph these are tasks with no prerequisites.

        Returns
        -------
        list of str

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D')])
        >>> g.roots()
        ['A']
        """
        return sorted(n for n in self._forward_adj if not self._reverse_adj[n])

    def leaves(self):
        """
        Nodes with no outgoing arrows (sinks / outputs).
        In a dependency graph these are the final products.

        Returns
        -------
        list of str

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D')])
        >>> g.leaves()
        ['D']
        """
        return sorted(n for n in self._forward_adj if not self._forward_adj[n])

    # ── core algorithms ────────────────────────────────────────────────

    def topological_sort(self):
        """
        ITERATIVE — Kahn's algorithm (BFS over in-degree counts).
        Return nodes in an order where every node appears only AFTER
        all the nodes it depends on.  Ties broken alphabetically.

        Returns
        -------
        list of str

        Example
        -------
        >>> g = DAG(edges=[('design','backend'), ('design','frontend'),
        ...                ('backend','tests'), ('frontend','tests'),
        ...                ('tests','deploy')])
        >>> g.topological_sort()
        ['design', 'backend', 'frontend', 'tests', 'deploy']
        """
        in_deg = {n: self.in_degree(n) for n in self._forward_adj}
        queue  = sorted(n for n in in_deg if in_deg[n] == 0)
        result = []

        if self._debug:
            print(f"\n[DAG.topological_sort]  Kahn's algorithm")
            print("  in-degree = how many arrows arrive at each node?")
            print("  Nodes with in-degree 0 can start immediately.")
            print()
            print("  in-degree table:")
            for nd in sorted(in_deg):
                print(f"    {nd}: {in_deg[nd]}")
            print(f"\n  Initial queue (in-degree-0 nodes): {queue}")
            print()

        step = 0
        while queue:
            node = queue.pop(0)
            result.append(node)
            step += 1
            if self._debug:
                print(f"  Step {step}: process '{node}'  -> result: {result}")
            for child in sorted(self._forward_adj[node]):
                in_deg[child] -= 1
                if self._debug:
                    print(f"    decrement in-deg['{child}']: -> {in_deg[child]}"
                          + (" -> add to queue!" if in_deg[child] == 0 else ""))
                if in_deg[child] == 0:
                    queue.append(child)
                    queue.sort()

        if self._debug:
            print(f"\n  Final order: {' -> '.join(result)}")
        return result

    def descendants(self, node):
        """
        RECURSIVE — simple DFS following arrows forward.
        Return all nodes reachable FROM node (node itself excluded).

        Parameters
        ----------
        node : str

        Returns
        -------
        list of str (sorted)

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D')])
        >>> g.descendants('A')
        ['B', 'C', 'D']
        >>> g.descendants('B')
        ['D']
        """
        visited = set()

        def dfs(u):
            for v in self._forward_adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dfs(v)

        dfs(node)
        result = sorted(visited)

        if self._debug:
            print(f"\n[DAG.descendants]  reachable from '{node}': {result}")
        return result

    def ancestors(self, node):
        """
        RECURSIVE — DFS on the reverse graph (_reverse_adj).
        Return all nodes that can reach node (node itself excluded).

        Parameters
        ----------
        node : str

        Returns
        -------
        list of str (sorted)

        Example
        -------
        >>> g = DAG(edges=[('A','C'), ('B','C'), ('C','D')])
        >>> g.ancestors('D')
        ['A', 'B', 'C']
        >>> g.ancestors('A')   # nothing points to A
        []
        """
        visited = set()

        def dfs(u):
            for v in self._reverse_adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    dfs(v)

        dfs(node)
        result = sorted(visited)

        if self._debug:
            print(f"\n[DAG.ancestors]  can reach '{node}': {result}  (DFS on _reverse_adj)")
        return result

    def all_paths(self, src, dst):
        """
        RECURSIVE — DFS with backtracking.
        Return every directed simple path from src to dst.

        Parameters
        ----------
        src : str
        dst : str

        Returns
        -------
        list of list[str]   e.g. [['A','B','D'], ['A','C','D']]

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D')])
        >>> g.all_paths('A', 'D')
        [['A', 'B', 'D'], ['A', 'C', 'D']]
        >>> g.all_paths('B', 'A')   # no upstream path in a DAG
        []
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

        if self._debug:
            print(f"\n[DAG.all_paths]  '{src}' -> '{dst}'  ({len(result)} paths):")
            for p in result:
                print(f"  {' -> '.join(p)}")

        return result

    def shortest_path(self, src, dst):
        """
        ITERATIVE — BFS (list used as FIFO queue).
        Return the path using the fewest edges from src to dst, or []
        if no directed path exists.

        Parameters
        ----------
        src : str
        dst : str

        Returns
        -------
        list of str   e.g. ['A', 'B', 'D']  or  []

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D'), ('C','E')])
        >>> g.shortest_path('A', 'D')
        ['A', 'B', 'D']
        >>> g.shortest_path('D', 'A')   # no reverse path
        []
        """
        if src not in self._forward_adj or dst not in self._forward_adj:
            return []

        if self._debug:
            print(f"\n[DAG.shortest_path]  '{src}' -> '{dst}'  (BFS)")
            print("  BFS guarantees: first arrival at dst = fewest edges.")

        queue   = [[src]]
        visited = {src}

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == dst:
                if self._debug:
                    print(f"  Shortest path ({len(path)-1} edges): {' -> '.join(path)}")
                return path
            for child in self._forward_adj.get(node, []):
                if child not in visited:
                    visited.add(child)
                    queue.append(path + [child])

        if self._debug:
            print(f"  No path from '{src}' to '{dst}'")
        return []

    def longest_path(self, src, dst):
        """
        ITERATIVE — DP over topological order, O(V + E).
        Return the path using the most edges from src to dst, or []
        if no directed path exists.

        This is easy on a DAG but NP-hard on general graphs with cycles.
        Topological order guarantees dist[u] is finalised before we relax
        u's outgoing edges.

        Parameters
        ----------
        src : str
        dst : str

        Returns
        -------
        list of str   e.g. ['A', 'C', 'D']  or  []

        Example
        -------
        >>> g = DAG(edges=[('A','B'), ('A','C'), ('B','D'), ('C','D')])
        >>> g.longest_path('A', 'D')     # A->B->D and A->C->D are both length 2
        ['A', 'B', 'D']
        >>> g.longest_path('B', 'A')     # no reverse path
        []
        """
        topo = self.topological_sort()
        dist = {n: -1   for n in topo}
        prev = {n: None for n in topo}
        dist[src] = 0

        if self._debug:
            print(f"\n[DAG.longest_path]  '{src}' -> '{dst}'  (DP over topo order)")
            print("  dist[v] = max edges from src to v  (-1 = not reached yet)")
            print(f"  Topo order: {topo}")
            print(f"  Initial: dist['{src}']=0, rest=-1")
            print()

        for u in topo:
            if dist[u] == -1:
                continue
            for v in self._forward_adj.get(u, []):
                if dist[u] + 1 > dist[v]:
                    dist[v] = dist[u] + 1
                    prev[v] = u
                    if self._debug:
                        print(f"  Relax '{u}'->'{v}': dist['{v}'] = {dist[v]}")

        if dist.get(dst, -1) == -1:
            if self._debug:
                print(f"  '{dst}' unreachable -> []")
            return []

        path, cur = [], dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        if self._debug:
            print(f"\n  Longest path ({len(path)-1} edges): {' -> '.join(path)}")
        return path


# ====================================================================== #
#  Tree  —  rooted directed tree (one parent per non-root node)           #
# ====================================================================== #

class Tree(DAG):
    """
    Rooted directed tree.
    Every non-root node has EXACTLY ONE parent.
    Raises ValueError if that constraint is violated.

    Example
    -------
    >>> t = Tree(root='root', edges=[('root','A'),('root','B'),('A','C')])
    >>> t = Tree(root='f5', edges=[('f5','f4'),('f5','f3'),('f4','f2')])
    """

    def __init__(self, root, edges=None, debug=False):
        """
        Parameters
        ----------
        root  : str             the root node name (string)
        edges : list of (str, str) tuples   (parent, child) pairs
        debug : bool (default False)

        Raises
        ------
        ValueError   if any non-root node has != 1 parent

        Example
        -------
        >>> t = Tree(root='A', edges=[('A','B'), ('A','C'), ('B','D')])
        >>> t.root
        'A'
        """
        super().__init__(edges=edges, nodes=[root], debug=debug)

        for n in self.nodes():
            if n != root and self.in_degree(n) != 1:
                raise ValueError(
                    f"Node {n!r} has in-degree {self.in_degree(n)}"
                    f" — every non-root must have exactly one parent.")
        self.root = root

        if self._debug:
            print(f"\n[Tree.__init__]  Validation")
            print(f"  Root: '{root}' (in-degree=0)")
            for n in sorted(self.nodes()):
                if n != root:
                    print(f"  '{n}': in-degree={self.in_degree(n)}"
                          f"  parent='{self.parent(n)}'  depth={self.depth(n)}")
            print(f"  Tree height: {self.height()}")

    # ── accessors ─────────────────────────────────────────────────────

    def parent(self, node):
        """
        The unique parent of node, or None for the root.

        Parameters
        ----------
        node : str

        Returns
        -------
        str or None

        Example
        -------
        >>> t = Tree(root='A', edges=[('A','B'), ('B','C')])
        >>> t.parent('C')
        'B'
        >>> t.parent('A')   # root has no parent
        None
        """
        p = self.parents(node)
        return p[0] if p else None

    def is_leaf(self, node):
        """
        True when node has no children (is at the bottom of the tree).

        Parameters
        ----------
        node : str

        Returns
        -------
        bool

        Example
        -------
        >>> t = Tree(root='A', edges=[('A','B'), ('A','C')])
        >>> t.is_leaf('B')
        True
        >>> t.is_leaf('A')
        False
        """
        return self.out_degree(node) == 0

    def siblings(self, node):
        """
        Other children of the same parent (empty list for the root).

        Parameters
        ----------
        node : str

        Returns
        -------
        list of str (sorted)

        Example
        -------
        >>> t = Tree(root='R', edges=[('R','A'),('R','B'),('R','C'),('A','D')])
        >>> t.siblings('B')
        ['A', 'C']
        >>> t.siblings('R')   # root has no parent -> no siblings
        []
        """
        par = self.parent(node)
        if par is None:
            return []
        return sorted(c for c in self.children(par) if c != node)

    def depth(self, node):
        """
        RECURSIVE.
        Distance from the root down to node (root has depth 0).

        Parameters
        ----------
        node : str

        Returns
        -------
        int

        Example
        -------
        >>> t = Tree(root='A', edges=[('A','B'), ('B','C'), ('B','D')])
        >>> t.depth('A')
        0
        >>> t.depth('B')
        1
        >>> t.depth('C')
        2
        """
        if node == self.root:
            return 0
        return 1 + self.depth(self.parent(node))

    def height(self):
        """
        RECURSIVE — DFS from root.
        Depth of the deepest leaf (0 for a single-node tree).
        More efficient than max(depth(n) for leaves) because it
        visits each node once.

        Returns
        -------
        int

        Example
        -------
        >>> Tree(root='A', edges=[('A','B'),('B','C')]).height()
        2
        >>> Tree(root='A').height()   # single node
        0
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
        All nodes exactly d steps below the root (sorted).

        Parameters
        ----------
        d : int   desired depth (0 = root only)

        Returns
        -------
        list of str

        Example
        -------
        >>> t = Tree(root='R', edges=[('R','A'),('R','B'),('A','C'),('B','D')])
        >>> t.nodes_at_depth(0)
        ['R']
        >>> t.nodes_at_depth(1)
        ['A', 'B']
        >>> t.nodes_at_depth(2)
        ['C', 'D']
        """
        result = []

        def dfs(u, cur):
            if cur == d:
                result.append(u)
                return
            for v in self.children(u):
                dfs(v, cur + 1)

        dfs(self.root, 0)
        return sorted(result)

    def subtree(self, node):
        """
        RECURSIVE — DFS.
        Return a new Tree rooted at node containing node and all its
        descendants.  The original tree is not modified.

        Parameters
        ----------
        node : str   must be a node in this tree

        Returns
        -------
        Tree

        Example
        -------
        >>> t = Tree(root='R', edges=[('R','A'),('R','B'),('A','C'),('A','D')])
        >>> t.subtree('A').nodes()
        ['A', 'C', 'D']
        >>> t.subtree('A').root
        'A'
        """
        edges = []

        def dfs(u):
            for v in self.children(u):
                edges.append((u, v))
                dfs(v)

        dfs(node)
        return Tree(root=node, edges=edges, debug=self._debug)

    def lca(self, a, b):
        """
        Lowest Common Ancestor.
        The deepest node that is an ancestor of BOTH a and b
        (or a/b itself when one is a direct ancestor of the other).

        Parameters
        ----------
        a : str   first node
        b : str   second node

        Returns
        -------
        str or None   (None only when a and b are in disconnected trees)

        Example
        -------
        >>> t = Tree(root='root', edges=[('root','A'),('root','B'),
        ...                              ('A','C'),('A','D'),('B','E')])
        >>> t.lca('C', 'D')    # C and D share parent A
        'A'
        >>> t.lca('C', 'E')    # C is under A, E is under B -> meet at root
        'root'
        >>> t.lca('root', 'C') # root is an ancestor of C
        'root'
        """
        path_a = set()
        node   = a
        while node is not None:
            path_a.add(node)
            node = self.parent(node)

        node = b
        while node is not None:
            if node in path_a:
                if self._debug:
                    print(f"\n[Tree.lca]  lca('{a}','{b}') = '{node}'")
                    print(f"  Ancestors of '{a}': {sorted(path_a)}")
                    print(f"  Walk up from '{b}' until crossing that set: '{node}'")
                return node
            node = self.parent(node)

        return None

    def pre_order(self):
        """
        RECURSIVE — DFS, root first (parent visited BEFORE its children).
        Matches the order in which a recursive function is CALLED.

        Returns
        -------
        list of str

        Example
        -------
        >>> t = Tree(root='A', edges=[('A','B'),('A','C'),('B','D')])
        >>> t.pre_order()
        ['A', 'B', 'D', 'C']
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
        RECURSIVE — DFS, leaves first (parent visited AFTER all its children).
        Matches the order in which a recursive function RETURNS.

        Returns
        -------
        list of str

        Example
        -------
        >>> t = Tree(root='A', edges=[('A','B'),('A','C'),('B','D')])
        >>> t.post_order()
        ['D', 'B', 'C', 'A']
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
    and return to the starting node with the least total distance.

    Parameters
    ----------
    graph     : Graph                   any undirected Graph (usually complete)
    distances : dict of {(str,str): number}
                edge weights; symmetric — both (a,b) and (b,a) are looked up.
                e.g. {('A','B'): 5, ('B','C'): 3, ('A','C'): 7}
    start     : str (optional)          starting node; default = first alphabetical

    Returns
    -------
    (float, list of str)   (min_cost, path)
    path includes the return to start, e.g. ['A','B','C','A']

    Complexity: O(n!) — practical only for small n (≤ 10 or so).

    Example
    -------
    >>> cities = Graph(edges=[('A','B'),('A','C'),('B','C')])
    >>> d = {('A','B'): 2, ('B','C'): 3, ('A','C'): 9}
    >>> tsp(cities, d, start='A')
    (14, ['A', 'B', 'C', 'A'])
    """
    nodes = graph.nodes()
    if not nodes:
        return 0, []
    if start is None:
        start = nodes[0]

    def dist(a, b):
        return distances.get((a, b)) or distances.get((b, a), float('inf'))


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


def build_hanoi_tree(n, src="A", dst="C", via="B", debug=False):
    """
    RECURSIVE — build the call tree for Tower of Hanoi(n discs).

    Each node represents one recursive call: 'h(discs,src>dst)#id'.
    Pre-order  = order calls are MADE  (push onto call stack).
    Post-order = order calls RETURN (leaves = actual disc moves).

    Parameters
    ----------
    n   : int   number of discs, e.g.  build_hanoi_tree(3)
    src : str   source peg name         default 'A'
    dst : str   destination peg         default 'C'
    via : str   intermediate peg        default 'B'

    Returns
    -------
    Tree

    Example
    -------
    >>> t = build_hanoi_tree(3)
    >>> len(t.nodes())    # 2^3 - 1 = 7
    7
    >>> t.height()        # 0-indexed depth of deepest leaf
    2
    >>> len(t.leaves())   # 4 actual disc moves
    4
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
            rec(n - 1, src, via, dst, label)
            rec(n - 1, via, dst, src, label)

    rec(n, src, dst, via, None)
    return Tree(root=root[0], edges=edges, debug=debug)


# ====================================================================== #
#  __main__  —  five example problems                                     #
# ====================================================================== #

if __name__ == "__main__":
    import sys
    DEBUG = "--debug" in sys.argv
    if DEBUG:
        print("[debug=True] All graphs will print internal data structures.\n")
    else:
        print("[debug=False] Run with --debug for step-by-step explanations.\n")

    SEP = "=" * 62

    # 1. Fibonacci ──────────────────────────────────────────────────────
    print(SEP)
    print("1. FIBONACCI — subproblem DAG")
    print(SEP)
    fib = DAG(edges=[
        ("f5","f4"),("f5","f3"),("f4","f3"),("f4","f2"),
        ("f3","f2"),("f3","f1"),("f2","f1"),("f2","f0"),
    ])
    order  = fib.topological_sort()
    shared = [n for n in fib.nodes() if fib.in_degree(n) > 1]
    print("Evaluation order:", " -> ".join(order))
    print("Shared subproblems (in-degree > 1):", shared)
    print("has_path f5->f0 :", fib.has_path("f5", "f0"))
    print("all paths f5->f0:", fib.all_paths("f5", "f0"))

    # 2. Tower of Hanoi ─────────────────────────────────────────────────
    print()
    print(SEP)
    print("2. TOWER OF HANOI — recursive call Tree (n=3)")
    print(SEP)
    ht  = build_hanoi_tree(n=3, debug=DEBUG)
    pre = ht.pre_order()
    post = ht.post_order()
    print(f"Nodes: {len(ht.nodes())} (= 2^n-1)")
    print(f"Height: {ht.height()}  (root depth 0; leaves at depth n-1=2)")
    print(f"LCA of first and last leaf: {ht.lca(pre[-1], pre[1])}")
    print()
    print("Pre-order (call sequence):")
    for node in pre:
        d = ht.depth(node)
        print(f"  {'  '*d}-> {node}")
    print()
    print("Leaf nodes = actual disc moves (post-order):")
    for node in post:
        if ht.is_leaf(node):
            print(f"  {node}")

    # 3. Task Scheduling ────────────────────────────────────────────────
    print()
    print(SEP)
    print("3. TASK SCHEDULING — DAG")
    print(SEP)
    tasks = DAG(edges=[
        ("design","backend"),("design","frontend"),
        ("backend","tests"),("frontend","tests"),("tests","deploy"),
    ])
    print("Valid schedule  :", " -> ".join(tasks.topological_sort()))
    print("Critical path   :", " -> ".join(tasks.longest_path("design","deploy")))
    print("Shortest path   :", " -> ".join(tasks.shortest_path("design","deploy")))
    print("All paths       :", tasks.all_paths("design","deploy"))
    print("Ancestors/deploy:", tasks.ancestors("deploy"))

    # 4. Sensor Coverage ────────────────────────────────────────────────
    print()
    print(SEP)
    print("4. SENSOR NETWORK COVERAGE — Graph")
    print(SEP)
    locaties = Graph(edges=[("R","Q"),("U","Z"),("K","Y"),("Q","Z"),("L","K")])
    comps     = locaties.connected_components()
    solutions = locaties.minimum_dominating_sets()
    print(f"Nodes          : {locaties.nodes()}")
    print(f"Connected?     : {locaties.is_connected()}  -> two islands")
    print(f"Components     : {comps}")
    print(f"Min relay size : {len(solutions[0])}")
    print("All optimal sets:")
    for s in format_solutions(solutions, sep=", "):
        print(f"  {{ {s} }}")
    print()
    print("has_path K->Z  :", locaties.has_path("K", "Z"))
    print("all_paths R->Z :", locaties.all_paths("R", "Z"))

    # 5. Traveling Salesman ─────────────────────────────────────────────
    print()
    print(SEP)
    print("5. TRAVELING SALESMAN — backtracking on complete Graph")
    print(SEP)
    cities = Graph(edges=[("A","B"),("A","C"),("A","D"),("B","C"),("B","D"),("C","D")])
    d = {("A","B"):2,("A","C"):9,("A","D"):10,("B","C"):6,("B","D"):4,("C","D"):3}
    cost, path = tsp(cities, d, start="A")
    print(f"Cities         : {cities.nodes()}")
    print(f"Optimal tour   : {' -> '.join(path)}")
    print(f"Total distance : {cost}")
    print("  -> A->B(2)+B->D(4)+D->C(3)+C->A(9) = 18")

    # Cycle / SCC demo ──────────────────────────────────────────────────
    print()
    print(SEP)
    print("CYCLE DETECTION + SCC demo")
    print(SEP)
    dg = DirectedGraph(edges=[("A","B"),("B","C"),("C","A"),("B","D")], debug=DEBUG)
    print("A->B->C->A (+ B->D)")
    print("  has_cycle():", dg.has_cycle())
    print("  SCCs       :", dg.strongly_connected_components())
    print("  transpose  :", dg.transpose().edges())
    try:
        DAG(edges=[("X","Y"),("Y","Z"),("Z","X")])
    except ValueError as e:
        print(f"\nDAG(X->Y->Z->X): {e}")
    try:
        AcyclicGraph(edges=[("P","Q"),("Q","R"),("R","P")])
    except ValueError as e:
        print(f"AcyclicGraph(P-Q-R-P): {e}")

    # Tree extras ───────────────────────────────────────────────────────
    print()
    print(SEP)
    print("TREE extras — subtree, lca, nodes_at_depth, siblings")
    print(SEP)
    t = Tree(root="root", edges=[
        ("root","A"),("root","B"),("A","C"),("A","D"),("B","E"),
    ], debug=DEBUG)
    print("height()          :", t.height())
    print("nodes_at_depth(2) :", t.nodes_at_depth(2))
    print("lca(C, E)         :", t.lca("C","E"))
    print("lca(C, D)         :", t.lca("C","D"))
    print("siblings(C)       :", t.siblings("C"))
    print("subtree(A).nodes():", t.subtree("A").nodes())
    print("is_leaf(C)        :", t.is_leaf("C"))
    print("pre_order()       :", t.pre_order())
    print("post_order()      :", t.post_order())

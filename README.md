# Graph Theory Toolkit

📐 **Educational Python graph library — no external dependencies**

Five graph types built from plain Python dicts, lists and tuples, with classic algorithms and five toy problems.  
Written to be readable by anyone with a year of Python experience.

---

## Class hierarchy

```
Graph                  undirected, may have cycles
  └── AcyclicGraph     undirected, no cycles  (= forest)

DirectedGraph          directed, may have cycles
  └── DAG              directed, no cycles
        └── Tree       rooted, one parent per non-root node
```

## Methods at a glance

| Class | Method | What it answers |
|---|---|---|
| **Graph** | `nodes()` · `edges()` · `neighbors()` · `degree()` | basic structure |
| | `has_path(src,dst)` | can you get from A to B? |
| | `is_connected()` | is the whole graph one piece? |
| | `connected_components()` | which groups of nodes form separate islands? |
| | `all_paths(src,dst)` | every possible route A→B |
| | `minimum_dominating_sets()` | smallest relay-station set (NP-complete) |
| **DirectedGraph** | inherits Graph methods + | |
| | `children()` · `parents()` · `in_degree()` · `out_degree()` | directional structure |
| | `has_cycle()` | is there a directed loop? |
| | `transpose()` | flip every arrow |
| | `strongly_connected_components()` | groups where every node reaches every other (Kosaraju) |
| **DAG** | inherits DirectedGraph + | |
| | `roots()` · `leaves()` | entry/exit points |
| | `topological_sort()` | order respecting all dependencies (Kahn's BFS) |
| | `descendants()` · `ancestors()` | transitive reachability |
| | `all_paths(src,dst)` | every directed route |
| | `shortest_path(src,dst)` | fewest-edge route (BFS) |
| | `longest_path(src,dst)` | most-edge route (DP over topo order) |
| **Tree** | inherits DAG + | |
| | `parent()` · `depth()` · `height()` · `is_leaf()` | tree position |
| | `nodes_at_depth(d)` | all nodes d steps from root |
| | `subtree(node)` | extract sub-tree below node |
| | `siblings(node)` · `lca(a,b)` | tree relationships |
| | `pre_order()` · `post_order()` | DFS traversals |

### Standalone algorithms

| Function | Description |
|---|---|
| `tsp(graph, distances, start)` | Traveling Salesman — shortest Hamiltonian cycle (backtracking) |
| `build_hanoi_tree(n)` | Tower of Hanoi recursive call tree |
| `format_solutions(solutions, sep)` | Pretty-print solution lists |

---

## Five example problems (`python graphs.py`)

| # | Problem | Graph type | Algorithm |
|---|---|---|---|
| 1 | Fibonacci DP | `DAG` | `topological_sort` — shared subproblems |
| 2 | Tower of Hanoi | `Tree` | `pre_order` / `post_order` — call sequence vs return sequence |
| 3 | Task scheduling | `DAG` | `topological_sort` + `longest_path` (critical path) |
| 4 | Sensor coverage | `Graph` | `minimum_dominating_sets` + `connected_components` |
| 5 | Traveling Salesman | `Graph` + `tsp()` | recursive backtracking |

---

## Quick start

```bash
# No installation needed — plain Python 3.9+
python graphs.py
```

### Generate PDF

```bash
pip install fpdf2
python generate_pdf.py          # writes graphs.pdf  (~14 pages A4)
```

---

## Files

| File | Description |
|---|---|
| [`graphs.py`](graphs.py) | All five graph classes + algorithms + `__main__` examples |
| [`dag.py`](dag.py) | Standalone `DAG` class with extended API (kept for reference) |
| [`dag_problems.ipynb`](dag_problems.ipynb) | Jupyter notebook with `draw_dag()` visualiser + 5 problems |
| [`create_notebook.py`](create_notebook.py) | Script that regenerates the notebook from source |
| [`generate_pdf.py`](generate_pdf.py) | Converts `graphs.py` to a syntax-coloured A4 PDF |

---

## Algorithm complexity

| Method | Algorithm | O |
|---|---|---|
| `topological_sort` | Kahn's BFS | O(V+E) |
| `shortest_path` | BFS | O(V+E) |
| `longest_path` | DP over topological order | O(V+E) |
| `descendants` / `ancestors` | DFS | O(V+E) |
| `has_cycle` | DFS three-colour | O(V+E) |
| `strongly_connected_components` | Kosaraju (2×DFS) | O(V+E) |
| `minimum_dominating_sets` | Backtracking + pruning | NP-complete |
| `tsp` | Backtracking | O(n!) |

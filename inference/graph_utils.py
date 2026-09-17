import networkx as nx
import numpy as np

def parse_swc_to_graph(swc_path):
    """
    Reads an SWC file and returns a NetworkX graph.
    """
    G = nx.Graph()
    with open(swc_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 7:
                n, type_id, x, y, z, radius, parent = parts
                n = int(n)
                parent = int(parent)
                G.add_node(n, pos=np.array([float(z), float(y), float(x)]), type_id=int(type_id), radius=float(radius))
                if parent != -1:
                    G.add_edge(n, parent)
    return G

def extract_endpoints_and_trees(G):
    """
    Separates the graph into connected components (trees).
    Finds endpoints (degree == 1) for each tree.
    Returns:
        trees: list of NetworkX subgraphs
        endpoints: list of tuples (tree_idx, node_id, pos)
    """
    trees = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    endpoints = []
    for t_idx, tree in enumerate(trees):
        for node in tree.nodes():
            if tree.degree(node) <= 1:  # Endpoints or isolated nodes
                pos = tree.nodes[node]['pos']
                endpoints.append((t_idx, node, pos))
                
    return trees, endpoints

def get_endpoints_in_window(endpoints, window, margin=0):
    """
    Filters endpoints that fall within the specified 3D window.
    window = (z_start, y_start, x_start, z_end, y_end, x_end)
    """
    zs, ys, xs, ze, ye, xe = window
    
    in_window = []
    for ep in endpoints:
        t_idx, n_id, pos = ep
        z, y, x = pos
        if (zs - margin <= z <= ze + margin) and \
           (ys - margin <= y <= ye + margin) and \
           (xs - margin <= x <= xe + margin):
            in_window.append(ep)
            
    return in_window

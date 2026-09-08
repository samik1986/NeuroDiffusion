import networkx as nx
import numpy as np

def parse_swc_to_graph(swc_path):
    """
    Parses an SWC file into a NetworkX directed graph.
    Assumes standard SWC format: ID, Type, X, Y, Z, Radius, Parent
    """
    G = nx.DiGraph()
    with open(swc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 7:
                n_id = int(parts[0])
                n_type = int(parts[1])
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                radius = float(parts[5])
                parent = int(parts[6])
                
                G.add_node(n_id, type=n_type, pos=np.array([x, y, z]), radius=radius)
                if parent != -1:
                    G.add_edge(parent, n_id)
    return G

def separate_trees(swc_graph):
    """
    Divides the SWC graph into multiple isolated sub-graphs (trees).
    Returns a list of NetworkX weakly connected components.
    """
    # Use weakly connected components to extract disconnected trees
    trees = [swc_graph.subgraph(c).copy() for c in nx.weakly_connected_components(swc_graph)]
    return trees

def get_tree_endpoints(tree):
    """
    Returns nodes in a tree that are endpoints (degree == 1 in undirected sense).
    In directed SWC, these are roots (in_degree=0) and leaves (out_degree=0).
    """
    endpoints = []
    for node in tree.nodes():
        if tree.in_degree(node) == 0 or tree.out_degree(node) == 0:
            endpoints.append(node)
    return endpoints

def find_nearest_neighbors(trees, max_distance=10.0):
    """
    For each tree, finds the nearest endpoint in other trees within max_distance.
    Returns a list of candidate pairs: (tree_i_idx, node_i, tree_j_idx, node_j, distance)
    """
    candidates = []
    num_trees = len(trees)
    
    # Precompute endpoints and their positions for fast lookup
    tree_endpoints = {}
    for i, tree in enumerate(trees):
        endpoints = get_tree_endpoints(tree)
        positions = {n: tree.nodes[n]['pos'] for n in endpoints}
        tree_endpoints[i] = {'nodes': endpoints, 'pos': positions}
        
    for i in range(num_trees):
        for j in range(i + 1, num_trees):
            eps_i = tree_endpoints[i]['nodes']
            eps_j = tree_endpoints[j]['nodes']
            
            for n_i in eps_i:
                pos_i = tree_endpoints[i]['pos'][n_i]
                for n_j in eps_j:
                    pos_j = tree_endpoints[j]['pos'][n_j]
                    
                    dist = np.linalg.norm(pos_i - pos_j)
                    if dist <= max_distance:
                        candidates.append((i, n_i, j, n_j, dist))
                        
    # Sort candidates by distance
    candidates.sort(key=lambda x: x[4])
    return candidates

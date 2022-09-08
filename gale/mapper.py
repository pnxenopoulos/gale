import gudhi as gd
import itertools
import kmapper as km
import multiprocessing
import networkx as nx
import numpy as np

from sklearn.cluster import AgglomerativeClustering


def create_mapper(
    X: np.ndarray,
    f: np.ndarray,
    resolution: int,
    gain: float,
    dist_thresh: float,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
) -> dict:
    """Runs Mapper on given some data, a filter function, and resolution + gain parameters.

    Args:
        X (np.ndarray): Array of data. For GALE, this is the feature attribution output (n x k), where there are n samples with k feature attributions each.
        f (np.ndarray): Filter (lens) function. For GALE, the predicted probabilities are the lens function.
        resolution (int): Resolution (how wide each window is)
        gain (float): Gain (how much overlap between windows)
        dist_thresh (float): If using AgglomerativeClustering, this sets the distance threshold as (X.max() - X.min())*thresh. Ignored if clusterer is not AgglomerativeClustering
        clusterer (sklearn.base.ClusterMixin, optional): Clustering method from sklearn. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").

    Returns:
        dict: Dictionary containing the Mapper output
    """
    mapper = km.KeplerMapper(verbose=0)
    cover = km.Cover(resolution, gain)
    clusterer.distance_threshold = (X.max() - X.min()) * dist_thresh
    graph = mapper.map(lens=f, X=X, clusterer=clusterer, cover=cover)
    graph["node_attr"] = {}
    for cluster in graph["nodes"]:
        graph["node_attr"][cluster] = np.mean(f[graph["nodes"][cluster]])
    return graph


def create_pd(mapper: dict) -> list:
    """Creates a persistence diagram from Mapper output.

    Args:
        mapper (dict): Mapper output from `create_mapper`

    Returns:
        list: List of the topographical features
    """
    st = gd.SimplexTree()
    node_idx = {}
    for i, n in enumerate(mapper["nodes"].keys()):
        node_idx[n] = i
        st.insert([i])
    for origin in mapper["links"]:
        edges = mapper["links"][origin]
        for e in edges:
            if e != origin:
                st.insert([node_idx[origin], node_idx[e]])
    attrs = {node_idx[k]: mapper["node_attr"][k] for k in mapper["nodes"].keys()}
    for k, v in attrs.items():
        st.assign_filtration([k], v)
    st.make_filtration_non_decreasing()
    st.extend_filtration()
    dgms = st.extended_persistence(min_persistence=1e-5)
    pdgms = []
    for dgm in dgms:
        pdgms += [d[1] for d in dgm]
    return pdgms


def bottleneck_distance(mapper_a: dict, mapper_b: dict) -> float:
    """Calculates the bottleneck distance between two Mapper outputs (denoted A and B)

    Args:
        mapper_a (dict): Mapper A, from `create_mapper`
        mapper_b (dict): Mapper B, from `create_mapper`

    Returns:
        float: the bottleneck distance
    """
    pd_a = create_pd(mapper_a)
    pd_b = create_pd(mapper_b)
    return gd.bottleneck_distance(pd_a, pd_b)


# Sub function to run the bootstrap sequence
def _bootstrap_sub(params):
    M = create_mapper(
        X=params[0],
        f=params[1],
        resolution=params[2],
        gain=params[3],
        dist_thresh=params[4],
        clusterer=params[5],
    )
    n_samples = params[0].shape[0]
    distribution, cc = [], []
    for bootstrap in range(params[7]):
        # Randomly select points with replacement
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        Xboot = params[0][idxs, :]
        fboot = params[1][idxs]
        # Fit mapper
        M_boot = create_mapper(Xboot, fboot, params[2], params[3], params[4], params[5])
        G_boot = mapper_to_networkx(M_boot)
        G_cc = nx.number_connected_components(G_boot)
        cc.append(G_cc)
        distribution.append(bottleneck_distance(M_boot, M))
    distribution = np.sort(distribution)
    dist_thresh = distribution[int(params[6] * len(distribution))]
    cc = np.sort(cc)
    cc_thresh = cc[int(params[6] * len(cc))]
    return params[2], params[3], params[4], dist_thresh, cc_thresh


def bootstrap_mapper_params(
    X: np.ndarray,
    f: np.ndarray,
    resolutions: list,
    gains: list,
    distances: list,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
    ci=0.95,
    n=30,
    n_jobs=1,
) -> dict:
    """Bootstraps the data to figure out the best Mapper parameters through a greedy search.

    Args:
        X (np.ndarray): Array of data. For GALE, this is the feature attribution output (n x k), where there are n samples with k feature attributions each.
        f (np.ndarray): Filter (lens) function. For GALE, the predicted probabilities are the lens function.
        resolutions (list): List of resolutions to test.
        gains (list): List of gains to test.
        distances (list): If using AgglomerativeClustering, this sets the distance threshold as (X.max() - X.min())*thresh.
        clusterer (sklearn.base.ClusterMixin, optional): Clustering method from sklearn. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").
        ci (float, optional): Confidence interval to create. Defaults to 0.95.
        n (int, optional): Number of bootstraps to run. Defaults to 30.
        n_jobs (int, optional): Number of processes for multiprocessing. Defaults to CPU count. -1 for all cores.

    Returns:
        dict: Dictionary containing the Mapper parameters found in a greedy search
    """
    # Create parameter list
    paramlist = list(
        itertools.product(
            [X], [f], resolutions, gains, distances, [clusterer], [ci], [n]
        )
    )

    # Create MP pool
    if n_jobs < 1:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(processes=n_jobs)

    results = pool.map(_bootstrap_sub, paramlist)

    # Find "best" parameters by scaling stability and components between [0,1]
    # then, calculate the distance to (0,0). Select the params with minimum distance.
    best_stability = None
    best_component = None
    best_r = None
    best_g = None
    best_d = None

    # Put stabilities, components, etc. into lists
    stability, component, resolution, gain, distance = [], [], [], [], []
    for res in results:
        stability.append(res[3])
        component.append(res[4])
        resolution.append(res[0])
        gain.append(res[1])
        distance.append(res[2])

    # Find min/max for stability and components, for scaling purposes
    min_distance = 999
    min_stability = min(stability)
    max_stability = max(stability)
    min_component = min(component)
    max_component = max(component)

    # Calculate distance to (0,0) and take the smallest
    for s, c, r, g, d in zip(stability, component, resolution, gain, distance):
        stab = (s - min_stability) / (max_stability - min_stability)
        comp = (c - min_component) / (max_component - min_component)
        dist = np.sqrt(stab**2 + comp**2)
        if dist < min_distance:
            min_distance = dist
            best_stability = s
            best_component = c
            best_r = r
            best_g = g
            best_d = d

    return {
        "stability": best_stability,
        "components": best_component,
        "resolution": best_r,
        "gain": best_g,
        "distance_threshold": best_d,
    }


def mapper_to_networkx(mapper: dict) -> nx.classes.graph.Graph:
    """Takes the Mapper output (which is a `dict`) and transforms it to a networkx graph.

    Args:
        mapper (dict): Mapper output from `create_mapper`

    Returns:
        nx.classes.graph.Graph: Networkx graph produced by the Mapper output.
    """
    G = nx.Graph()
    node_idx = {}
    for i, n in enumerate(mapper["nodes"].keys()):
        node_idx[n] = i
        G.add_node(i)
    for origin in mapper["links"]:
        edges = mapper["links"][origin]
        for e in edges:
            if e != origin:
                G.add_edge(node_idx[origin], node_idx[e])
    attrs = {
        node_idx[k]: {"avg_pred": mapper["node_attr"][k]}
        for k in mapper["nodes"].keys()
    }
    nx.set_node_attributes(G, attrs)
    return G

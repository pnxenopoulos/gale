import gudhi as gd
import kmapper as km
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


def bootstrap_mapper_params(
    X: np.ndarray,
    f: np.ndarray,
    resolutions: list,
    gains: list,
    distances: list,
    clusterer=AgglomerativeClustering(n_clusters=None, linkage="single"),
    ci=0.95,
    n=30,
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

    Returns:
        dict: Dictionary containing the Mapper parameters found in a greedy search
    """
    bootstrapped_results = {}
    for r in resolutions:
        bootstrapped_results[r] = {}
        for g in gains:
            bootstrapped_results[r][g] = {}
            for d in distances:
                bootstrapped_results[r][g][d] = {}
                M = create_mapper(X, f, r, g, d, clusterer)
                n_samples = X.shape[0]
                distribution, cc = [], []
                for bootstrap in range(n):
                    # Randomly select points with replacement
                    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
                    Xboot = X[idxs, :]
                    fboot = f[idxs]
                    # Fit mapper
                    M_boot = create_mapper(Xboot, fboot, r, g, d, clusterer)
                    G_boot = mapper_to_networkx(M_boot)
                    G_cc = nx.number_connected_components(G_boot)
                    cc.append(G_cc)
                    distribution.append(bottleneck_distance(M_boot, M))
                distribution = np.sort(distribution)
                dist_thresh = distribution[int(ci * len(distribution))]
                cc = np.sort(cc)
                cc_thresh = cc[int(ci * len(cc))]
                bootstrapped_results[r][g][d]["stability"] = dist_thresh
                bootstrapped_results[r][g][d]["components"] = cc_thresh
    # Find good params
    best_stability = 999
    best_components = 999
    best_r = None
    best_g = None
    best_d = None
    for r in resolutions:
        for g in gains:
            for d in distances:
                if (bootstrapped_results[r][g][d]["stability"] < best_stability) & (
                    bootstrapped_results[r][g][d]["components"] < best_components
                ):
                    best_stability = bootstrapped_results[r][g][d]["stability"]
                    best_components = bootstrapped_results[r][g][d]["components"]
                    best_r = r
                    best_g = g
                    best_d = d
    return {
        "stability": best_stability,
        "components": best_components,
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

import gudhi as gd
import networkx as nx
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn_tda import MapperComplex



def generate_mapper(
    X: np.ndarray, f: np.ndarray, f_bounds: list, resolution: int, gain: float, inp="point cloud", clustering=AgglomerativeClustering(n_clusters=None, linkage="single")
) -> sklearn_tda.clustering.MapperComplex:
    """Generate Mapper object.

    Args:
        X (np.ndarray): Explanations
        f (np.ndarray): Filter function
        f_bounds (list): Filter function bounds
        resolution (int): Resolution
        gain (float): Gain
        inp (str, optional): [description]. Defaults to "point cloud".
        clustering ([type], optional): [description]. Defaults to AgglomerativeClustering(n_clusters=None, linkage="single").

    Returns:
        sklearn_tda.clustering.MapperComplex: Mapper graph
    """
    clustering.distance_threshold = (X.max() - X.min())*0.5
    params = {
        "filters": f, 
        "filter_bnds": np.array([[0,1]]), 
        "colors": f, 
        "resolutions": np.array([resolution]), 
        "gains": np.array([gain]), 
        "inp": inp, 
        "clustering": clustering
    }
    M = MapperComplex(**params).fit(X)
    return M


def bottleneck_distance(mapper_a, mapper_b):
    """Computes the bottleneck distance between two Mapper graphs.

    Args:
        mapper_a (sklearn_tda.clustering.MapperComplex): Mapper graph
        mapper_b (sklearn_tda.clustering.MapperComplex): Mapper graph

    Returns:
        float: A float indicating the bottleneck distance between two Mapper graphs.
    """
    pd_a = get_persistence_diagram(mapper_a)
    pd_b = get_persistence_diagram(mapper_b)
    return gd.bottleneck_distance(pd_a, pd_b)


def mapper_to_nx(mapper, get_attrs=False):
    """Turn a Mapper graph (as computed by sklearn_tda) into a networkx graph. Taken from https://github.com/MathieuCarriere/statmapper/blob/master/statmapper/statmapper.py

    Args:
        mapper (sklearn_tda.clustering.MapperComplex): A graph computed by Mapper
        get_attrs (bool, optional): Use the Mapper attributes or not. Defaults to False.
    """
    M = mapper.mapper_
	G = nx.Graph()
	for (splx,_) in M.get_skeleton(1):	
		if len(splx) == 1:	G.add_node(splx[0])
		if len(splx) == 2:	G.add_edge(splx[0], splx[1])
	if get_attrs:
		attrs = {k: {"attr_name": mapper.node_info_[k]["colors"]} for k in G.nodes()}
		nx.set_node_attributes(G, attrs)
	return G

def get_persistence_diagram(mapper):
    """Gets the persistence diagram for a Mapper graph

    Args:
        mapper (sklearn_tda.clustering.MapperComplex): A graph computed by Mapper

    Returns:
        list: [description]
    """
    st = compute_persistence_diagram(mapper)
    st.extend_filtration()
    dgms = st.extended_persistence(min_persistence=1e-5)
    pdgms = []
    for dgm in dgms:
        pdgms += [d[1] for d in dgm]
    return pdgms


def compute_persistence_diagram(mapper):
    """Computes the persistence diagram. Used for get_persistence_diagram.

    Args:
        mapper (sklearn_tda.clustering.MapperComplex): A graph computed by Mapper

    Returns:
        gudhi.SimplexTree: [description]
    """
    G = nx.Graph()
    M = mapper.mapper_
    st = gd.SimplexTree()
    for (splx,_) in M.get_skeleton(1):
        if len(splx) == 1:  G.add_node(splx[0])
        if len(splx) == 2:  G.add_edge(splx[0], splx[1])
    attrs = {k: mapper.node_info_[k]["colors"][0] for k in G.nodes()}
    for n in G.nodes():
        st.insert([n])
    for e1,e2 in G.edges():
        st.insert([e1,e2])
    for k,v in attrs.items():
        st.assign_filtration([k],v)
    st.make_filtration_non_decreasing()
    return st

import numpy as np

from sklearn_tda import MapperComplex


def generate_mapper(
    X: np.ndarray, f: np.ndarray, res: int, gain: float
) -> sklearn_tda.clustering.MapperComplex:
    """Computes the MAPPER graph given a f, resolution and gain.

    Args:
        X (np.ndarray): Data
        f (np.ndarray): f
        res (int): Resolution
        gain (float): Gain

    Returns:
        sklearn_tda.clustering.MapperComplex: Mapper object
    """
    pass


def bottleneck_distance():
    pass


def mapper_to_nx():
    pass


def get_persistence_diagram():
    pass


def compute_persistence_diagram():
    pass

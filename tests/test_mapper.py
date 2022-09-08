import pytest
import numpy as np
import networkx as nx

from gale import (
    create_mapper,
    create_pd,
    bottleneck_distance,
    bootstrap_mapper_params,
    mapper_to_networkx,
)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer


class TestMapper:
    """Class to test Mapper functions"""

    def setup_class(self):
        """Create data used for tests"""
        self.X, self.y = make_classification(
            n_samples=1000, n_features=10, random_state=2022
        )
        self.clf = LogisticRegression(random_state=2022).fit(self.X, self.y)
        self.preds = self.clf.predict_proba(self.X)[:, 1]

        lime_exp = []
        explainer = LimeTabularExplainer(
            self.X, mode="classification", discretize_continuous=False
        )

        for i in range(0, self.X.shape[0]):
            exp = explainer.explain_instance(self.X[i, :], self.clf.predict_proba)
            tmp = [0 for i in range(self.X.shape[1])]
            for e in exp.as_list():
                tmp[int(e[0])] = e[1]
            lime_exp.append(tmp)

        self.explanations = np.array(lime_exp)
        self.M = create_mapper(self.explanations, self.preds, 10, 0.3, 0.5)

    def test_mapper(self):
        """Tests Mapper creation"""
        assert type(self.M) == dict
        assert "nodes" in self.M.keys()
        assert "links" in self.M.keys()
        assert "node_attr" in self.M.keys()

    def test_pd(self):
        """Tests persistence diagram"""
        self.PD = create_pd(self.M)
        assert type(self.PD) == list
        assert type(self.PD[0]) == tuple
        assert len(self.PD[0]) == 2

    def test_bottleneck(self):
        """Tests bottleneck distance"""
        self.BD = bottleneck_distance(self.M, self.M)
        assert pytest.approx(self.BD) == 0

    def test_bootstrap(self):
        """Tests bootstrap parameter finding"""
        self.PARAMS = bootstrap_mapper_params(
            self.X, self.preds, [10, 20, 30], [0.2, 0.3, 0.4], [0.2, 0.3, 0.4]
        )
        print(self.PARAMS)
        assert type(self.PARAMS) == dict
        assert type(self.PARAMS["resolution"]) == int
        assert type(self.PARAMS["gain"]) == float
        assert type(self.PARAMS["distance_threshold"]) == float
        assert self.PARAMS["resolution"] == 10
        assert self.PARAMS["gain"] == 0.2
        assert self.PARAMS["distance_threshold"] == 0.4

    def test_mapper_to_nx(self):
        """Tests Mapper to networkx conversion"""
        self.G = mapper_to_networkx(self.M)
        assert type(self.G) == nx.classes.graph.Graph
        assert len(self.G.nodes()) == 10

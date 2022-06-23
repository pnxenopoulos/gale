from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(1000, n_features=10)
lr = LogisticRegression()
lr.fit(X, y)
preds = lr.predict_proba(X)[:, 1]

from lime.lime_tabular import LimeTabularExplainer

lime_exp = []
explainer = LimeTabularExplainer(X, mode="classification", discretize_continuous=False)

for i in range(0, X.shape[0]):
    exp = explainer.explain_instance(X[i, :], lr.predict_proba)
    tmp = [0 for i in range(X.shape[1])]
    for e in exp.as_list():
        tmp[int(e[0])] = e[1]
    lime_exp.append(tmp)

import numpy as np

lime_exp = np.array(lime_exp)

m = create_mapper(lime_exp, preds, 10, 0.3)

cl = AgglomerativeClustering(n_clusters=None, linkage="single")
cl.distance_threshold = (lime_exp.max() - lime_exp.min()) * 0.5

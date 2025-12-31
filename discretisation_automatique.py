import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import kruskal

warnings.simplefilter(action='ignore', category=FutureWarning)

def decoup(base, x, y, h=0, k=0, pAUC=0, nbmod=3, calcul=1, algo='Nelder-Mead', graphe=0):
    Xt = base[x].dropna()
    Yt = base[y].dropna()

    X = Xt
    Y = Yt

    seuils = np.quantile(X, np.linspace(0, 1, nbmod+1))[1:nbmod]

    def fitauc(s):
        s2 = np.concatenate(([-np.inf], np.unique(s), [np.inf]))
        qX = pd.cut(X, bins=s2)
        logit = LogisticRegression(solver='liblinear').fit(pd.get_dummies(qX, drop_first=True), Y)
        qXn = logit.predict_proba(pd.get_dummies(pd.cut(base[x], bins=s2), drop_first=True))[:, 1]
        auc_value = roc_auc_score(Y, qXn)
        prop_table = pd.crosstab(qX, Y, normalize='index')
        result = (auc_value * (1 - np.sum((pd.value_counts(qX, normalize=True) ** 2))) / 
                 (1 - (1-h) * (np.sum((pd.value_counts(qX, normalize=True) ** 2)))) * 
                 ((1 - (1-k) * (np.sum(prop_table.iloc[:, 1] ** 2))) / 
                 (1 - np.sum(prop_table.iloc[:, 1] ** 2))))
        return -result

    def applical():
        sf = np.concatenate(([-np.inf], est.x, [np.inf]))
        qX = pd.cut(Xt, bins=sf)
        tab = pd.crosstab(pd.cut(Xt, bins=sf), Yt)
        print("\nResult of cutting :\n")
        print("\nTresholds      % Negative  % Positive   # +   #   % #")
        print(pd.concat([tab.div(tab.sum(axis=1), axis=0) * 100, tab.iloc[:, 1], tab.sum(axis=1), tab.sum(axis=1) * 100 / len(Xt)], axis=1))
        print("\nConvergence indicator (0 = convergence optimisation)")
        print(est.success) 
        print("\nMaximum (partial) AUC  :")
        print(-est.fun)
        print("\nClass homogeneity (0 <- low ... high -> 1) :")
        prop_table = pd.crosstab(qX, Yt, normalize='index')
        print(1 - np.sum(prop_table.iloc[:, 1] ** 2))
        return qX

    def gini(t):
        print("\nAUC before cutting :")
        logit = LogisticRegression(solver='liblinear').fit(X.values.reshape(-1, 1), Y)
        g1 = roc_auc_score(Y, logit.predict_proba(X.values.reshape(-1, 1))[:, 1])
        print(g1)
        print("\nAUC after cutting :")
        logit = LogisticRegression(solver='liblinear').fit(pd.get_dummies(t, drop_first=True), Yt)
        g2 = roc_auc_score(Yt, logit.predict_proba(pd.get_dummies(t, drop_first=True))[:, 1])
        print(g2)
        print("\n% Evolution AUC before/after cutting :")
        print(100 * (g2 - g1) / g1)
        print("\n")

    if calcul == 1:
        est = minimize(fitauc, seuils, method=algo)
    else:
        est = {'x': bornes, 'success': True, 'fun': 0}

    print("\n---------------------------------------------------------------------------")
    print(f"\nDiscretization of {x} into {nbmod} classes (algorithm {algo})")
    print("\n---------------------------------------------------------------------------")

    qX1 = applical()
    gini(qX1)

    if graphe == 1:
        base0 = X[Y == 0]
        base1 = X[Y == 1]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.hist(base0, bins=30, density=True, alpha=0.5, color="blue", label=f"{y} = 0")
        plt.hist(base1, bins=30, density=True, alpha=0.5, color="red", label=f"{y} = 1")
        plt.axvline(np.median(base0), color="blue", linestyle="--")
        plt.axvline(np.median(base1), color="red", linestyle="--")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.boxplot([base0, base1], vert=False)
        plt.yticks([1, 2], [f"{y} = 0", f"{y} = 1"])
        
        plt.tight_layout()
        plt.show()

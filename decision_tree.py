"""
Árvore de Decisão C4.5 implementada 
Suporta critérios 'entropy' (Gain Ratio) e 'gini' (Gini Impurity).
"""

import math
from collections import Counter

import numpy as np


class DecisionTreeC45:
    """
    Árvore de Decisão estilo C4.5 com Gain Ratio (entropy) ou Gini Impurity.

    Parâmetros
    ----------
    max_depth : int ou None
        Profundidade máxima da árvore. None = sem limite.
    min_samples_split : int
        Número mínimo de amostras para tentar dividir um nó.
    criterion : str
        'entropy' → Gain Ratio; 'gini' → Gini impurity.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy'):
        if criterion not in ('entropy', 'gini'):
            raise ValueError(f"criterion deve ser 'entropy' ou 'gini', recebido: {criterion!r}")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None          # raiz da árvore (estrutura interna)
        self.classes_ = None      # classes conhecidas no treino
        self.n_features_ = None   # número de features

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Constrói a árvore de decisão.

        Parâmetros
        ----------
        X : lista de listas (array-like)
            Dados de treino, shape (n_amostras, n_features).
        y : lista (array-like)
            Rótulos de treino, shape (n_amostras,).

        Retorna
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if len(X) == 0:
            raise ValueError("X não pode estar vazio")
        if len(y) == 0:
            raise ValueError("y não pode estar vazio")
        if len(X) != len(y):
            raise ValueError(f"X e y têm tamanhos diferentes: {len(X)} vs {len(y)}")

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        """
        Prediz a classe de cada amostra.

        Parâmetros
        ----------
        X : lista de listas (array-like)
            Dados a classificar, shape (n_amostras, n_features).

        Retorna
        -------
        lista de predições (mesmo comprimento de X).
        """
        if self.tree is None:
            raise RuntimeError("O modelo ainda não foi treinado – chame fit() primeiro.")
        X = np.asarray(X, dtype=float)
        return [self._predict_one(row, self.tree) for row in X]

    # ------------------------------------------------------------------
    # Predição de uma amostra (recursiva)
    # ------------------------------------------------------------------

    def _predict_one(self, x, node):
        """Percorre a árvore até uma folha e retorna a classe armazenada."""
        if node['is_leaf']:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    # ------------------------------------------------------------------
    # Cálculos de impureza
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(y):
        """Entropia de Shannon: H(S) = -Σ p(c)·log2(p(c))."""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _gini(y):
        """Gini impurity: Gini(S) = 1 - Σ p(c)²."""
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return float(1.0 - np.sum(probs ** 2))

    # ------------------------------------------------------------------
    # Ganho de informação (redução de impureza)
    # ------------------------------------------------------------------

    def _impurity(self, y):
        """Retorna a impureza de acordo com o critério escolhido."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        else:
            return self._gini(y)

    def _information_gain(self, y_parent, y_left, y_right):
        """
        IG = impureza(pai) - [ (nL/n)·impureza(esq) + (nR/n)·impureza(dir) ].
        """
        n = len(y_parent)
        nL, nR = len(y_left), len(y_right)
        if nL == 0 or nR == 0:
            return 0.0
        parent_imp = self._impurity(y_parent)
        child_imp = (nL / n) * self._impurity(y_left) + (nR / n) * self._impurity(y_right)
        return parent_imp - child_imp

    @staticmethod
    def _split_information(y_left, y_right, n_total):
        """
        Split Information: SI = - Σ (|Sv|/|S|)·log2(|Sv|/|S|).
        Para split binário: -(pL·log2(pL) + pR·log2(pR)).
        """
        nL, nR = len(y_left), len(y_right)
        if nL == 0 or nR == 0:
            return 0.0
        pL = nL / n_total
        pR = nR / n_total
        si = 0.0
        if pL > 0:
            si -= pL * math.log2(pL)
        if pR > 0:
            si -= pR * math.log2(pR)
        return si

    def _gain_ratio(self, y_parent, y_left, y_right):
        """Gain Ratio = Information Gain / Split Information."""
        ig = self._information_gain(y_parent, y_left, y_right)
        si = self._split_information(y_left, y_right, len(y_parent))
        if si == 0.0:
            return 0.0
        return ig / si

    def _score_split(self, y_parent, y_left, y_right):
        """
        Retorna a métrica de qualidade do split:
          - entropy  → Gain Ratio (C4.5)
          - gini     → Gini Gain   (CART)
        Quanto maior, melhor.
        """
        if self.criterion == 'entropy':
            return self._gain_ratio(y_parent, y_left, y_right)
        else:
            return self._information_gain(y_parent, y_left, y_right)

    # ------------------------------------------------------------------
    # Busca do melhor split
    # ------------------------------------------------------------------

    def _best_split(self, X, y):
        """
        Varre todas as features e pontos de corte possíveis.
        Retorna (feature_idx, threshold, score) ou (None, None, -inf).
        """
        n_samples, n_features = X.shape
        best_score = -float('inf')
        best_feature = None
        best_threshold = None

        for feat_idx in range(n_features):
            values = X[:, feat_idx]
            unique_vals = np.unique(values)

            if len(unique_vals) <= 1:
                continue  # feature constante, não dá para dividir

            # Pontos médios entre valores ordenados consecutivos
            sorted_vals = np.sort(unique_vals)
            midpoints = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0

            for thr in midpoints:
                left_mask = values <= thr
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                score = self._score_split(y, y_left, y_right)

                if score > best_score:
                    best_score = score
                    best_feature = feat_idx
                    best_threshold = thr

        return best_feature, best_threshold, best_score

    # ------------------------------------------------------------------
    # Construção recursiva da árvore
    # ------------------------------------------------------------------

    def _build_tree(self, X, y, depth):
        """
        Constrói recursivamente a árvore.
        Retorna um dicionário representando o nó.
        """
        n_samples = len(y)

        # --- critérios de parada ---

        # Nó vazio (não deve ocorrer em condições normais)
        if n_samples == 0:
            return {'is_leaf': True, 'class': None}

        # Todos da mesma classe
        unique = np.unique(y)
        if len(unique) == 1:
            return {'is_leaf': True, 'class': unique[0]}

        # Profundidade máxima atingida
        if self.max_depth is not None and depth >= self.max_depth:
            return {'is_leaf': True, 'class': self._majority(y)}

        # Poucas amostras para dividir
        if n_samples < self.min_samples_split:
            return {'is_leaf': True, 'class': self._majority(y)}

        # --- busca do melhor split ---
        feat, thr, score = self._best_split(X, y)

        # Nenhum split útil encontrado
        if feat is None or score <= 0.0:
            return {'is_leaf': True, 'class': self._majority(y)}

        # --- divisão ---
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'is_leaf': False,
            'feature': int(feat),
            'threshold': float(thr),
            'left': left_child,
            'right': right_child,
            'class': self._majority(y),   # classe majoritária neste nó
        }

    @staticmethod
    def _majority(y):
        """Retorna a classe mais frequente em y."""
        return Counter(y).most_common(1)[0][0]


# ======================================================================
# Testes simples
# ======================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TESTE 1 – Dataset sintético simples (2 classes, 2 features)")
    print("=" * 60)

    # Duas features contínuas, duas classes
    X_train = [
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 3.0],
        [6.0, 7.0],
        [7.0, 8.0],
        [8.0, 8.0],
    ]
    y_train = [0, 0, 0, 1, 1, 1]

    for crit in ('entropy', 'gini'):
        print(f"\n--- criterion = {crit!r} ---")
        tree = DecisionTreeC45(criterion=crit, max_depth=3)
        tree.fit(X_train, y_train)
        preds = tree.predict(X_train)
        acc = sum(1 for p, t in zip(preds, y_train) if p == t) / len(y_train)
        print(f"Predições treino: {preds}")
        print(f"Acurácia treino : {acc:.2%}")

        # Teste em dados novos
        X_test = [[2.5, 3.5], [7.5, 7.5], [4.0, 4.0]]
        preds_test = tree.predict(X_test)
        print(f"Predições teste : {preds_test}")

    print("\n" + "=" * 60)
    print("TESTE 2 – Dataset IRIS-like (3 classes)")
    print("=" * 60)

    # Simulando algo parecido com Iris: 3 classes, 4 features
    np.random.seed(42)
    n_per_class = 10
    X_iris = []
    y_iris = []

    # Classe 0: centro (2,2), algum espalhamento
    X_iris.extend(np.random.normal(loc=2.0, scale=0.5, size=(n_per_class, 4)))
    y_iris.extend([0] * n_per_class)

    # Classe 1: centro (5,5)
    X_iris.extend(np.random.normal(loc=5.0, scale=0.5, size=(n_per_class, 4)))
    y_iris.extend([1] * n_per_class)

    # Classe 2: centro (8,8)
    X_iris.extend(np.random.normal(loc=8.0, scale=0.5, size=(n_per_class, 4)))
    y_iris.extend([2] * n_per_class)

    X_iris = np.array(X_iris)
    y_iris = np.array(y_iris)

    # Embaralhar
    idx = np.random.permutation(len(y_iris))
    X_iris, y_iris = X_iris[idx], y_iris[idx]

    # Split 70/30 manual
    split = int(0.7 * len(y_iris))
    X_tr, y_tr = X_iris[:split], y_iris[:split]
    X_te, y_te = X_iris[split:], y_iris[split:]

    for crit in ('entropy', 'gini'):
        tree = DecisionTreeC45(criterion=crit, max_depth=5, min_samples_split=2)
        tree.fit(X_tr, y_tr)

        preds_tr = tree.predict(X_tr)
        preds_te = tree.predict(X_te)

        acc_tr = sum(1 for p, t in zip(preds_tr, y_tr) if p == t) / len(y_tr)
        acc_te = sum(1 for p, t in zip(preds_te, y_te) if p == t) / len(y_te)

        print(f"\n--- criterion = {crit!r} ---")
        print(f"Acurácia treino: {acc_tr:.2%}")
        print(f"Acurácia teste : {acc_te:.2%}")

    print("\n" + "=" * 60)
    print("TESTE 3 – max_depth e min_samples_split")
    print("=" * 60)

    tree_stump = DecisionTreeC45(max_depth=1)
    tree_stump.fit(X_tr, y_tr)
    print(f"max_depth=1  → treino: {sum(1 for p, t in zip(tree_stump.predict(X_tr), y_tr) if p == t) / len(y_tr):.2%}")

    tree_big_min = DecisionTreeC45(min_samples_split=10)
    tree_big_min.fit(X_tr, y_tr)
    print(f"min_samples_split=10 → treino: {sum(1 for p, t in zip(tree_big_min.predict(X_tr), y_tr) if p == t) / len(y_tr):.2%}")

    print("\n✅ Todos os testes concluídos!")

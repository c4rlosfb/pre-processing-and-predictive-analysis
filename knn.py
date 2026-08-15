"""
knn.py
Implementação do classificador K-Nearest Neighbors (K-NN).
Compatível com grid_search de metricas.py (interface fit/predict padrão).
"""

import math
import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Classificador K-Nearest Neighbors.

        Parâmetros:
        - n_neighbors: int - número de vizinhos a considerar (default: 5)
        - weights: 'uniform' (todos vizinhos pesam igual) ou
                   'distance' (peso = 1 / (distância + 1e-10))
        """
        if weights not in ('uniform', 'distance'):
            raise ValueError(
                f"weights deve ser 'uniform' ou 'distance', recebido: {weights!r}"
            )
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Armazena os dados de treino.

        Parâmetros:
        - X: lista de listas (amostras × features) ou numpy array
        - y: lista de rótulos

        Retorna:
        - self (para encadeamento)
        """
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)
        return self

    def _distancia_euclidiana(self, a, b):
        """
        Calcula a distância euclidiana entre dois pontos.
        Implementação manual: raiz da soma dos quadrados das diferenças.

        Parâmetros:
        - a, b: arrays numpy 1D com as features

        Retorna:
        - float: distância euclidiana
        """
        soma = 0.0
        for i in range(len(a)):
            diff = a[i] - b[i]
            soma += diff * diff
        return math.sqrt(soma)

    def _distancias_para_todos(self, amostra):
        """
        Calcula a distância euclidiana da amostra para TODOS os pontos de treino.

        Parâmetros:
        - amostra: array numpy 1D com as features

        Retorna:
        - array numpy 1D com as distâncias para cada ponto de treino
        """
        n = len(self.X_train)
        dists = np.empty(n, dtype=np.float64)
        for i in range(n):
            dists[i] = self._distancia_euclidiana(amostra, self.X_train[i])
        return dists

    def _predizer_uma(self, amostra):
        """
        Prediz a classe de uma única amostra.

        Parâmetros:
        - amostra: array numpy 1D com as features

        Retorna:
        - rótulo predito
        """
        # 1. Distâncias para todos os pontos de treino
        distancias = self._distancias_para_todos(amostra)

        # 2. Ordenar por distância e pegar os k vizinhos mais próximos
        indices = np.argsort(distancias)
        k = min(self.n_neighbors, len(indices))
        idx_k = indices[:k]
        dists_k = distancias[idx_k]
        classes_k = self.y_train[idx_k]

        # 3. Votação
        if self.weights == 'uniform':
            return self._voto_uniforme(classes_k, dists_k)
        else:  # 'distance'
            return self._voto_ponderado(classes_k, dists_k)

    def _voto_uniforme(self, classes_vizinhos, distancias_vizinhos):
        """
        Voto majoritário: cada vizinho vota com peso igual.
        Em caso de empate, vence a classe do vizinho mais próximo
        (já que classes_vizinhos e distancias_vizinhos estão ordenados por distância).
        """
        contagem = Counter(classes_vizinhos)
        max_votos = max(contagem.values())
        candidatas = [c for c, v in contagem.items() if v == max_votos]

        if len(candidatas) == 1:
            return candidatas[0]

        # Desempate: primeira classe entre as empatadas que aparece
        # (vizinhos já estão em ordem crescente de distância)
        for classe in classes_vizinhos:
            if classe in candidatas:
                return classe

        return candidatas[0]  # fallback (nunca deveria chegar aqui)

    def _voto_ponderado(self, classes_vizinhos, distancias_vizinhos):
        """
        Voto ponderado pela distância: peso = 1 / (distância + 1e-10).
        Em caso de empate na soma dos pesos, vence a classe do vizinho mais próximo.
        """
        pesos = {}
        for classe, dist in zip(classes_vizinhos, distancias_vizinhos):
            peso = 1.0 / (dist + 1e-10)
            pesos[classe] = pesos.get(classe, 0.0) + peso

        max_peso = max(pesos.values())
        candidatas = [c for c, p in pesos.items() if p == max_peso]

        if len(candidatas) == 1:
            return candidatas[0]

        # Desempate: vizinho mais próximo entre as empatadas
        for classe in classes_vizinhos:
            if classe in candidatas:
                return classe

        return candidatas[0]  # fallback

    def predict(self, X):
        """
        Prediz classes para todas as amostras em X.

        Parâmetros:
        - X: lista de listas (amostras × features) ou numpy array

        Retorna:
        - lista de rótulos preditos
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Modelo não foi treinado. Chame fit(X, y) antes de predict()."
            )

        X_arr = np.array(X, dtype=np.float64)

        # Caso uma única amostra seja passada como lista 1D
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        predicoes = [self._predizer_uma(amostra) for amostra in X_arr]
        # Converter numpy scalars para tipos nativos do Python
        return [int(p) if isinstance(p, (np.integer,)) else p for p in predicoes]


# ================================================================
# TESTES
# ================================================================
if __name__ == '__main__':
    print("=" * 62)
    print(" TESTE DO KNNClassifier ".center(62, "="))
    print("=" * 62)

    # ----------------------------------------------------------
    # Dataset sintético: 3 features, classes 0 e 1
    # ----------------------------------------------------------
    X_treino = [
        [1.0, 2.0, 3.0],
        [1.5, 2.5, 3.5],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
        [5.5, 6.5, 7.5],
        [6.0, 7.0, 8.0],
        [1.2, 2.2, 3.2],
        [5.2, 6.2, 7.2],
    ]
    y_treino = [0, 0, 0, 1, 1, 1, 0, 1]

    X_teste = [
        [1.3, 2.3, 3.3],   # esperado: 0
        [5.3, 6.3, 7.3],   # esperado: 1
        [3.0, 4.0, 5.0],   # fronteira
    ]
    y_esperado = [0, 1, None]

    # --- Teste 1: uniform, k=3 ---
    print("\n[1] KNN uniforme, k=3")
    knn = KNNClassifier(n_neighbors=3, weights='uniform')
    knn.fit(X_treino, y_treino)
    preds = knn.predict(X_teste)
    for i, (p, e) in enumerate(zip(preds, y_esperado)):
        status = "✓" if e is None or p == e else "✗"
        esperado_str = f"(esperado {e})" if e is not None else ""
        print(f"    Amostra {i}: pred={p} {esperado_str} {status}")

    # --- Teste 2: distance, k=3 ---
    print("\n[2] KNN ponderado por distância, k=3")
    knn = KNNClassifier(n_neighbors=3, weights='distance')
    knn.fit(X_treino, y_treino)
    preds = knn.predict(X_teste)
    for i, (p, e) in enumerate(zip(preds, y_esperado)):
        status = "✓" if e is None or p == e else "✗"
        esperado_str = f"(esperado {e})" if e is not None else ""
        print(f"    Amostra {i}: pred={p} {esperado_str} {status}")

    # --- Teste 3: k=1 (vizinho mais próximo) ---
    print("\n[3] KNN k=1 (vizinho mais próximo)")
    knn = KNNClassifier(n_neighbors=1)
    knn.fit(X_treino, y_treino)
    preds = knn.predict(X_teste)
    for i, (p, e) in enumerate(zip(preds, y_esperado)):
        status = "✓" if e is None or p == e else "✗"
        esperado_str = f"(esperado {e})" if e is not None else ""
        print(f"    Amostra {i}: pred={p} {esperado_str} {status}")

    # --- Teste 4: k > n_treino (usa todos) ---
    print("\n[4] KNN k=20 (> n_treino=8) → usa todos os pontos")
    knn = KNNClassifier(n_neighbors=20)
    knn.fit(X_treino, y_treino)
    preds = knn.predict(X_teste)
    print(f"    Predições: {preds}")

    # --- Teste 5: compatibilidade com grid_search (**params) ---
    print("\n[5] Compatibilidade com grid_search (construtor **params)")
    params = {'n_neighbors': 3, 'weights': 'distance'}
    modelo = KNNClassifier(**params)
    modelo.fit(X_treino, y_treino)
    preds = modelo.predict(X_teste)
    print(f"    Predições: {preds}")

    # --- Teste 6: amostra única 1D ---
    print("\n[6] Amostra única como lista 1D")
    knn = KNNClassifier(n_neighbors=3)
    knn.fit(X_treino, y_treino)
    pred = knn.predict([1.0, 2.0, 3.0])
    print(f"    Predição: {pred} (esperado 0)")

    # --- Teste 7: desempate explícito ---
    print("\n[7] Desempate: vizinho mais próximo decide")
    # 2 classes, k=2, vizinhos de classes diferentes → empate
    X_emp = [[1.0, 1.0], [9.0, 9.0]]
    y_emp = [0, 1]
    knn = KNNClassifier(n_neighbors=2)
    knn.fit(X_emp, y_emp)
    pred = knn.predict([[2.0, 2.0]])  # mais próximo de [1,1] (classe 0)
    print(f"    Predição: {pred} (esperado 0 — vizinho mais próximo)")

    # --- Teste 8: erro ao prever sem fit ---
    print("\n[8] Erro ao prever sem fit()")
    knn = KNNClassifier()
    try:
        knn.predict([[1.0, 2.0]])
        print("    ✗ Deveria ter levantado erro!")
    except RuntimeError as e:
        print(f"    ✓ RuntimeError capturado: {e}")

    # --- Teste 9: weights inválido ---
    print("\n[9] Parâmetro weights inválido")
    try:
        KNNClassifier(weights='manhattan')
        print("    ✗ Deveria ter levantado erro!")
    except ValueError as e:
        print(f"    ✓ ValueError capturado: {e}")

    print("\n" + "=" * 62)
    print(" TODOS OS TESTES CONCLUÍDOS ".center(62, "="))
    print("=" * 62)

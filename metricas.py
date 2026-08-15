
import math
import numpy as np
from collections import Counter


# ================================================================
# MÉTRICAS DE CLASSIFICAÇÃO
# ================================================================

def matriz_confusao(y_true, y_pred, labels=None):
    """Matriz de confusão: [[VN, FP], [FN, VP]]"""
    if labels is None:
        labels = sorted(set(list(y_true) + list(y_pred)))
    
    n = len(labels)
    matriz = [[0] * n for _ in range(n)]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
    for t, p in zip(y_true, y_pred):
        matriz[label_to_idx[t]][label_to_idx[p]] += 1
    
    return matriz, labels


def acuracia(y_true, y_pred):
    """Acurácia: (VP + VN) / Total"""
    corretos = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return corretos / len(y_true) if y_true else 0.0


def precisao(y_true, y_pred, classe_positiva=1):
    """Precisão: VP / (VP + FP)"""
    vp = sum(1 for t, p in zip(y_true, y_pred) if t == classe_positiva and p == classe_positiva)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != classe_positiva and p == classe_positiva)
    return vp / (vp + fp) if (vp + fp) > 0 else 0.0


def revocacao(y_true, y_pred, classe_positiva=1):
    """Recall/Sensibilidade/Revocação: VP / (VP + FN)"""
    vp = sum(1 for t, p in zip(y_true, y_pred) if t == classe_positiva and p == classe_positiva)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == classe_positiva and p != classe_positiva)
    return vp / (vp + fn) if (vp + fn) > 0 else 0.0


def f1_score(y_true, y_pred, classe_positiva=1):
    """F1-Score: média harmônica de precisão e recall."""
    p = precisao(y_true, y_pred, classe_positiva)
    r = revocacao(y_true, y_pred, classe_positiva)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def relatorio_classificacao(y_true, y_pred, nomes_classes=None):
    """Relatório completo de classificação."""
    classes = sorted(set(list(y_true) + list(y_pred)))
    if nomes_classes is None:
        nomes_classes = {c: str(c) for c in classes}
    
    linhas = []
    for c in classes:
        nome = nomes_classes.get(c, str(c))
        p = precisao(y_true, y_pred, c)
        r = revocacao(y_true, y_pred, c)
        f1 = f1_score(y_true, y_pred, c)
        sup = sum(1 for t in y_true if t == c)
        linhas.append(f"  {nome:<15} {p:.4f}     {r:.4f}     {f1:.4f}     {sup}")
    
    acc = acuracia(y_true, y_pred)
    total = len(y_true)
    
    cabecalho = f"{'Classe':<15} {'Precisão':>8}  {'Recall':>8}  {'F1':>8}  {'Suporte':>8}"
    sep = "-" * 60
    
    relatorio = f"{cabecalho}\n{sep}\n"
    relatorio += "\n".join(linhas)
    relatorio += f"\n{sep}\n"
    relatorio += f"  {'Acurácia':<15} {acc:>8.4f}                          {total}\n"
    
    return relatorio


# ================================================================
# VALIDAÇÃO CRUZADA
# ================================================================

def stratified_kfold(y, n_splits=5, seed=42):
    """Gera índices de treino/teste para Stratified K-Fold.
    Retorna lista de (train_idx, test_idx)."""
    import random
    random.seed(seed)
    
    # Agrupar índices por classe
    classes = {}
    for i, classe in enumerate(y):
        classes.setdefault(classe, []).append(i)
    
    # Embaralhar dentro de cada classe
    for indices in classes.values():
        random.shuffle(indices)
    
    # Distribuir em folds mantendo proporção
    folds = [[] for _ in range(n_splits)]
    for classe, indices in classes.items():
        for j, idx in enumerate(indices):
            folds[j % n_splits].append(idx)
    
    # Embaralhar cada fold
    for fold in folds:
        random.shuffle(fold)
    
    # Gerar splits
    splits = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = []
        for j in range(n_splits):
            if j != i:
                train_idx.extend(folds[j])
        random.shuffle(train_idx)
        splits.append((train_idx, test_idx))
    
    return splits


def cross_val_score(modelo_fn, X, y, cv_splits, scoring_fn=None):
    """Executa validação cruzada.
    modelo_fn: função (X_train, y_train) -> modelo treinado com .predict(X)
    cv_splits: saída de stratified_kfold()
    scoring_fn: função (y_true, y_pred) -> score (default: acurácia)
    Retorna: lista de scores por fold."""
    if scoring_fn is None:
        scoring_fn = acuracia
    
    scores = []
    for train_idx, test_idx in cv_splits:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        
        modelo = modelo_fn(X_train, y_train)
        y_pred = modelo.predict(X_test)
        scores.append(scoring_fn(y_test, y_pred))
    
    return scores


# ================================================================
# GRID SEARCH
# ================================================================

def grid_search(modelo_class, param_grid, X, y, cv_splits, scoring_fn=None):
    """Grid Search manual.
    modelo_class: classe com construtor (**params) e método .fit(X, y)
    param_grid: dicionário {param: [valores]}
    cv_splits: saída de stratified_kfold()
    scoring_fn: função (y_true, y_pred) -> score
    Retorna: (best_params, best_score, all_results)"""
    if scoring_fn is None:
        scoring_fn = f1_score
    
    # Gerar todas as combinações de parâmetros
    import itertools
    chaves = list(param_grid.keys())
    valores = list(param_grid.values())
    combinacoes = list(itertools.product(*valores))
    
    best_score = -1.0
    best_params = None
    all_results = []
    
    for combo in combinacoes:
        params = dict(zip(chaves, combo))
        
        # Avaliar com CV
        scores = []
        for train_idx, test_idx in cv_splits:
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]
            
            modelo = modelo_class(**params)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            scores.append(scoring_fn(y_test, y_pred))
        
        score_medio = sum(scores) / len(scores)
        score_std = desvio_padrao_amostral(scores)
        
        all_results.append({'params': params, 'score_medio': score_medio, 
                           'score_std': score_std, 'scores': scores})
        
        if score_medio > best_score:
            best_score = score_medio
            best_params = params
    
    return best_params, best_score, all_results


def desvio_padrao_amostral(dados):
    """Desvio padrão amostral (sem usar funcs_preprocessamento para evitar circularidade)."""
    n = len(dados)
    if n < 2:
        return 0.0
    m = sum(dados) / n
    var = sum((x - m) ** 2 for x in dados) / (n - 1)
    return math.sqrt(var)


# ================================================================
# DIVISÃO DE DADOS SIMPLES
# ================================================================

def train_test_split_simples(X, y, test_size=0.2, seed=42):
    """Split simples (sem estratificação)."""
    import random
    random.seed(seed)
    
    indices = list(range(len(y)))
    random.shuffle(indices)
    
    n_test = max(1, int(len(y) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    
    return X_train, X_test, y_train, y_test

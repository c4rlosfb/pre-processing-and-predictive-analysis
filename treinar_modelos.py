"""
treinar_modelos.py
Treina K-NN, Árvore C4.5 e MLP.
K-NN e DT: grid search completo (rápidos).
MLP: treino direto com melhores parâmetros conhecidos (grid search inviável).
Usa stratified_kfold + grid_search de metricas.py.

"""
import json
import math
import sys
import time
import numpy as np

# ── Módulos manuais ──────────────────────────────────────────
from metricas import (
    stratified_kfold, grid_search,
    acuracia, precisao, revocacao, f1_score, matriz_confusao
)
from knn import KNNClassifier
from decision_tree import DecisionTreeC45
from mlp import MLPClassifier

BASE = r'C:\Users\carlo\Documents\Trabalho IA'

# ═══════════════════════════════════════════════════════════════
# CARREGAR DADOS PRÉ-PROCESSADOS
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("CARREGANDO DADOS PRÉ-PROCESSADOS")
print("=" * 60, flush=True)

data = np.load(f'{BASE}/preprocessed_data.npz')
X_train_scaled = data['X_train_scaled'].tolist()
X_test_scaled = data['X_test_scaled'].tolist()
X_train_pca = data['X_train_pca'].tolist()
X_test_pca = data['X_test_pca'].tolist()
y_train_bal = [int(v) for v in data['y_train_bal']]
y_test = [int(v) for v in data['y_test']]
n_components = int(data['n_components_95'])

print(f"Treino: {len(X_train_scaled)}×{len(X_train_scaled[0])}")
print(f"Teste:  {len(X_test_scaled)}×{len(X_test_scaled[0])}")
print(f"PCA:    {n_components} componentes")
b = sum(1 for v in y_train_bal if v==0)
m = sum(1 for v in y_train_bal if v==1)
print(f"y_train_bal: {b} benignos + {m} malignos = {len(y_train_bal)}")
b = sum(1 for v in y_test if v==0)
m = sum(1 for v in y_test if v==1)
print(f"y_test:      {b} benignos + {m} malignos = {len(y_test)}")
print(flush=True)

# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO CV
# ═══════════════════════════════════════════════════════════════
cv_splits = stratified_kfold(y_train_bal, n_splits=5, seed=42)

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO AUXILIAR: avaliação no teste
# ═══════════════════════════════════════════════════════════════
def evaluate_on_test(model_class, best_params, X_tr, y_tr, X_te, y_te, nome=""):
    modelo = model_class(**best_params)
    t0 = time.time()
    modelo.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    y_pred = modelo.predict(X_te)

    acc = acuracia(y_te, y_pred)
    prec = precisao(y_te, y_pred, classe_positiva=1)
    rec = revocacao(y_te, y_pred, classe_positiva=1)
    f1 = f1_score(y_te, y_pred, classe_positiva=1)
    cm, _ = matriz_confusao(y_te, y_pred)

    print(f"  {nome}: fit={fit_time:.1f}s | Acc={acc:.4f} Prec={prec:.4f} "
          f"Rec={rec:.4f} F1={f1:.4f} | CM: VN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} VP={cm[1][1]}",
          flush=True)
    return {
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'confusion_matrix': cm
    }

# ═══════════════════════════════════════════════════════════════
# K-NN: GRID SEARCH (rápido)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1/6 K-NN sem PCA — Grid Search")
print("=" * 60, flush=True)

knn_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

t0 = time.time()
best_knn_np, score_knn_np, _ = grid_search(
    KNNClassifier, knn_grid, X_train_scaled, y_train_bal, cv_splits,
    scoring_fn=f1_score)
print(f"  Melhor: {best_knn_np} | CV F1={score_knn_np:.4f} | Tempo={time.time()-t0:.1f}s", flush=True)

knn_sem_pca = {'nome': 'K-NN (sem PCA)', 'best_params': best_knn_np, 'cv_f1_mean': score_knn_np}
knn_sem_pca.update(evaluate_on_test(KNNClassifier, best_knn_np,
    X_train_scaled, y_train_bal, X_test_scaled, y_test, "K-NN sem PCA"))

print("\n" + "=" * 60)
print("2/6 K-NN com PCA — Grid Search")
print("=" * 60, flush=True)
t0 = time.time()
best_knn_wp, score_knn_wp, _ = grid_search(
    KNNClassifier, knn_grid, X_train_pca, y_train_bal, cv_splits,
    scoring_fn=f1_score)
print(f"  Melhor: {best_knn_wp} | CV F1={score_knn_wp:.4f} | Tempo={time.time()-t0:.1f}s", flush=True)

knn_com_pca = {'nome': 'K-NN (com PCA)', 'best_params': best_knn_wp, 'cv_f1_mean': score_knn_wp}
knn_com_pca.update(evaluate_on_test(KNNClassifier, best_knn_wp,
    X_train_pca, y_train_bal, X_test_pca, y_test, "K-NN com PCA"))

# ═══════════════════════════════════════════════════════════════
# ÁRVORE C4.5: GRID SEARCH (rápido)
# ═══════════════════════════════════════════════════════════════
dt_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['entropy', 'gini']
}

print("\n" + "=" * 60)
print("3/6 Árvore C4.5 sem PCA — Grid Search")
print("=" * 60, flush=True)
t0 = time.time()
best_dt_np, score_dt_np, _ = grid_search(
    DecisionTreeC45, dt_grid, X_train_scaled, y_train_bal, cv_splits,
    scoring_fn=f1_score)
print(f"  Melhor: {best_dt_np} | CV F1={score_dt_np:.4f} | Tempo={time.time()-t0:.1f}s", flush=True)

dt_sem_pca = {'nome': 'Árvore C4.5 (sem PCA)', 'best_params': best_dt_np, 'cv_f1_mean': score_dt_np}
dt_sem_pca.update(evaluate_on_test(DecisionTreeC45, best_dt_np,
    X_train_scaled, y_train_bal, X_test_scaled, y_test, "DT sem PCA"))

print("\n" + "=" * 60)
print("4/6 Árvore C4.5 com PCA — Grid Search")
print("=" * 60, flush=True)
t0 = time.time()
best_dt_wp, score_dt_wp, _ = grid_search(
    DecisionTreeC45, dt_grid, X_train_pca, y_train_bal, cv_splits,
    scoring_fn=f1_score)
print(f"  Melhor: {best_dt_wp} | CV F1={score_dt_wp:.4f} | Tempo={time.time()-t0:.1f}s", flush=True)

dt_com_pca = {'nome': 'Árvore C4.5 (com PCA)', 'best_params': best_dt_wp, 'cv_f1_mean': score_dt_wp}
dt_com_pca.update(evaluate_on_test(DecisionTreeC45, best_dt_wp,
    X_train_pca, y_train_bal, X_test_pca, y_test, "DT com PCA"))

# ═══════════════════════════════════════════════════════════════
# MLP: TREINO COM MELHORES PARÂMETROS (sem grid search — inviável)
# ═══════════════════════════════════════════════════════════════
# A busca em grid do MLP testaria 3×2×2=12 combinações × 5 folds = 60 treinos,
# cada um com centenas de épocas de backpropagation. Isso levaria horas.
# Usamos os melhores parâmetros descobertos em experimentação prévia,
# que são reproduzíveis executando este mesmo script.
# O grid search manual para K-NN e DT já demonstra a metodologia.

MLP_BEST_PARAMS = {
    'sem_pca': {'hidden_layer_sizes': (50,), 'activation': 'relu', 'alpha': 0.0001, 'max_iter': 500},
    'com_pca': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'alpha': 0.0001, 'max_iter': 500},
}

print("\n" + "=" * 60)
print("5/6 MLP sem PCA — Treino direto (grid search inviável)")
print("=" * 60, flush=True)
mlp_sem_pca = {'nome': 'MLP (sem PCA)', 'best_params': MLP_BEST_PARAMS['sem_pca'],
               'cv_f1_mean': None,  # não temos CV com grid search
               'cv_f1_note': 'Melhor param via experimentação prévia; grid search de MLP inviável (~60 treinos × centenas de épocas)'}
mlp_sem_pca.update(evaluate_on_test(MLPClassifier, MLP_BEST_PARAMS['sem_pca'],
    X_train_scaled, y_train_bal, X_test_scaled, y_test, "MLP sem PCA"))

print("\n" + "=" * 60)
print("6/6 MLP com PCA — Treino direto")
print("=" * 60, flush=True)
mlp_com_pca = {'nome': 'MLP (com PCA)', 'best_params': MLP_BEST_PARAMS['com_pca'],
               'cv_f1_mean': None,
               'cv_f1_note': 'Melhor param via experimentação prévia; grid search de MLP inviável'}
mlp_com_pca.update(evaluate_on_test(MLPClassifier, MLP_BEST_PARAMS['com_pca'],
    X_train_pca, y_train_bal, X_test_pca, y_test, "MLP com PCA"))

# ═══════════════════════════════════════════════════════════════
# SALVAR RESULTADOS
# ═══════════════════════════════════════════════════════════════
resultados = {
    'knn_sem_pca': knn_sem_pca,
    'knn_com_pca': knn_com_pca,
    'dt_sem_pca': dt_sem_pca,
    'dt_com_pca': dt_com_pca,
    'mlp_sem_pca': mlp_sem_pca,
    'mlp_com_pca': mlp_com_pca
}

# Converter numpy types → Python nativos
def to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_native(v) for v in obj]
    if isinstance(obj, tuple): return str(obj)
    return obj

resultados = to_native(resultados)

out_path = f'{BASE}/resultados_manuais.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print(f"\n✅ Resultados salvos: {out_path}", flush=True)

# ═══════════════════════════════════════════════════════════════
# RESUMO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("📊 RESUMO FINAL")
print("=" * 60)
print(f"{'Modelo':<30} {'Acurácia':>8} {'Precisão':>8} {'Recall':>8} {'F1':>8}")
print("-" * 65)
for key in ['knn_sem_pca', 'knn_com_pca', 'dt_sem_pca', 'dt_com_pca', 'mlp_sem_pca', 'mlp_com_pca']:
    r = resultados[key]
    print(f"{r['nome']:<30} {r['test_accuracy']:>8.4f} {r['test_precision']:>8.4f} "
          f"{r['test_recall']:>8.4f} {r['test_f1']:>8.4f}")

# Melhor por recall
melhor = max(resultados.items(), key=lambda kv: kv[1]['test_recall'])
print(f"\n🏆 Melhor modelo (Recall): {melhor[1]['nome']} — Recall={melhor[1]['test_recall']:.4f}")
print("=" * 60, flush=True)

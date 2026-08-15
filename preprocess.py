"""
Itens 1-14 do roteiro
Base: Breast Cancer Wisconsin (UCI ID 15)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import csv
import math

# ── Importar TODAS as funções manuais ──────────────────────────
from funcs_preprocessamento import (
    carregar_csv, extrair_coluna, extrair_matriz,
    media, mediana, moda, desvio_padrao, percentil, iqr,
    minimo, maximo, amplitude, skewness, kurtosis,
    train_test_split_estratificado, detectar_valores_ausentes,
    imputar_mediana, normalizar_zscore, aplicar_zscore,
    smote, pca, aplicar_pca, detectar_outliers_iqr,
    contar_duplicatas, valores_unicos, contar_valores, proporcoes
)

BASE = r'C:\Users\carlo\Documents\Trabalho IA'
os.chdir(BASE)

# ═══════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════
FEATURES = [
    'Clump_thickness', 'Uniformity_of_cell_size', 'Uniformity_of_cell_shape',
    'Marginal_adhesion', 'Single_epithelial_cell_size', 'Bare_nuclei',
    'Bland_chromatin', 'Normal_nucleoli', 'Mitoses'
]
COL_BARE = 5        # índice da coluna Bare_nuclei na matriz X
SEED = 42
TEST_SIZE = 0.2

# ═══════════════════════════════════════════════════════════════
# FUNÇÃO AUXILIAR: salvar CSV
# ═══════════════════════════════════════════════════════════════
def salvar_csv(caminho, dados, cabecalho=None):
    """Salva lista de listas (ou lista simples) como CSV."""
    with open(caminho, 'w', newline='') as f:
        writer = csv.writer(f)
        if cabecalho:
            writer.writerow(cabecalho)
        for linha in dados:
            if isinstance(linha, (list, tuple)):
                writer.writerow(linha)
            else:
                writer.writerow([linha])


# ═══════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("ITENS 1-3: Identificação do alvo, tipos e escalas")
print("=" * 60)

dados_brutos, cabecalho_csv = carregar_csv('breast_cancer_wisconsin.csv')
total = len(dados_brutos)
print(f"Shape: ({total}, {len(cabecalho_csv)})")

# Extrair X (colunas 0 a 8 = features) e y (coluna 9 = Class)
X_raw = extrair_matriz(dados_brutos, [0, 1, 2, 3, 4, 5, 6, 7, 8])
y_raw = []
for linha in dados_brutos:
    val = linha[9]
    if val == 2.0:
        y_raw.append(0)
    elif val == 4.0:
        y_raw.append(1)
    else:
        y_raw.append(0 if int(val) == 2 else 1)

print("Item 1 — Alvo: Class (2=benigno→0, 4=maligno→1)")
cont_y = contar_valores(y_raw)
prop_y = proporcoes(y_raw)
print(f"  Distribuição: {cont_y}")
print(f"  Proporção: {prop_y}")

print("Item 2 — Tipos: Todas features são quantitativas discretas (1-10)")
print("Item 3 — Escala: Ordinal — severidade crescente, distância não uniforme")

# Valores ausentes
ausentes_raw = detectar_valores_ausentes(X_raw)
print(f"\nValores ausentes (total): {len(ausentes_raw)}")
for j, nome in enumerate(FEATURES):
    n_nan = sum(1 for (lin, col) in ausentes_raw if col == j)
    if n_nan > 0:
        print(f"  {nome}: {n_nan} ausentes")


# ═══════════════════════════════════════════════════════════════
# 2. ITENS 4-6: Medidas estatísticas (nos dados brutos)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITENS 4-6: Medidas estatísticas")
print("=" * 60)

print("\n--- Item 4: Localidade ---")
for j, nome in enumerate(FEATURES):
    col = extrair_coluna(X_raw, j)
    print(f"  {nome}: média={media(col):.2f}, mediana={mediana(col):.1f}, "
          f"moda={moda(col)}")

print("\n--- Item 5: Espalhamento ---")
for j, nome in enumerate(FEATURES):
    col = extrair_coluna(X_raw, j)
    q1 = percentil(col, 25)
    q3 = percentil(col, 75)
    print(f"  {nome}: std={desvio_padrao(col):.2f}, "
          f"range=[{minimo(col)}-{maximo(col)}], IQR={q3 - q1:.1f}")

print("\n--- Item 6: Distribuição ---")
for j, nome in enumerate(FEATURES):
    col = extrair_coluna(X_raw, j)
    print(f"  {nome}: skew={skewness(col):.3f}, kurtosis={kurtosis(col):.3f}")


# ═══════════════════════════════════════════════════════════════
# 3. HISTOGRAMAS (matplotlib puro, sem seaborn)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, j in zip(axes.flat, range(len(FEATURES))):
    col = extrair_coluna(X_raw, j)
    ax.hist(col, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_title(FEATURES[j], fontsize=10)
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frequência')
plt.tight_layout()
plt.savefig('histogramas_features.png', dpi=100)
plt.close()
print("\n  → histogramas_features.png salvo")


# ═══════════════════════════════════════════════════════════════
# 4. ITEM 7: Train/Test split (estratificado)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 7: Separação train/test (80/20 estratificado)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split_estratificado(
    X_raw, y_raw, test_size=TEST_SIZE, seed=SEED
)

print(f"Treino: {len(X_train)} exemplos, Teste: {len(X_test)} exemplos")
print(f"Dist treino: {proporcoes(y_train)}")
print(f"Dist teste:  {proporcoes(y_test)}")


# ═══════════════════════════════════════════════════════════════
# 5. ITENS 8-10: Atributos, exemplos e amostragem
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITENS 8-10: Atributos, exemplos e amostragem")
print("=" * 60)
print("Item 8: Nenhum atributo removido — 9 features clinicamente relevantes")
dup = contar_duplicatas(X_raw)
print(f"Item 9: {dup} duplicatas — nenhuma remoção necessária")
print("Item 10: Amostragem não necessária — 699 registros suficientes")


# ═══════════════════════════════════════════════════════════════
# 6. ITEM 12d: Imputação (Bare_nuclei) — ANTES do SMOTE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 12d: Imputação Bare_nuclei (mediana)")
print("=" * 60)

# Contar NaN antes da imputação (na coluna Bare_nuclei)
col_bare_train_raw = extrair_coluna(X_train, COL_BARE)
col_bare_test_raw = extrair_coluna(X_test, COL_BARE)
n_missing_train = len(X_train) - len(col_bare_train_raw)
n_missing_test = len(X_test) - len(col_bare_test_raw)
print(f"Missing treino: {n_missing_train}")
print(f"Missing teste:  {n_missing_test}")

# Imputar treino (usa a mediana do próprio treino)
X_train_imp = imputar_mediana(X_train)

# Calcular mediana do treino para aplicar ao teste
med_bare_train = mediana(col_bare_train_raw)
print(f"Mediana (treino): {med_bare_train:.1f}")

# Imputar teste manualmente (só Bare_nuclei, usando mediana do treino)
X_test_imp = [list(linha) for linha in X_test]
for linha in X_test_imp:
    val = linha[COL_BARE]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        linha[COL_BARE] = med_bare_train

# Verificar pós-imputação
col_bare_train_pos = extrair_coluna(X_train_imp, COL_BARE)
col_bare_test_pos = extrair_coluna(X_test_imp, COL_BARE)
print(f"Missing pós (treino): {len(X_train_imp) - len(col_bare_train_pos)}")
print(f"Missing pós (teste):  {len(X_test_imp) - len(col_bare_test_pos)}")


# ═══════════════════════════════════════════════════════════════
# 7. ITEM 11: SMOTE — balanceamento (APÓS imputação)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 11: SMOTE — balanceamento das classes")
print("=" * 60)
print(f"Antes: {contar_valores(y_train)}")

X_train_bal, y_train_bal = smote(X_train_imp, y_train, k=5, seed=SEED)
# Converter np.int64 → Python int nativo (cosmético)
y_train_bal = [int(v) for v in y_train_bal]
print(f"Depois: {contar_valores(y_train_bal)}")


# ═══════════════════════════════════════════════════════════════
# 8. ITEM 12a: Outliers (IQR) no treino balanceado
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 12a: Outliers (IQR)")
print("=" * 60)

outliers_por_coluna = detectar_outliers_iqr(X_train_bal)
for j, nome in enumerate(FEATURES):
    print(f"  {nome}: {int(outliers_por_coluna[j])} outliers")
print("Decisão: NÃO remover — clinicamente informativos")


# ═══════════════════════════════════════════════════════════════
# 9. ITEM 12b-c: Inconsistências e redundâncias
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 12b-c: Inconsistências e redundâncias")
print("=" * 60)

# Range check (1-10) para cada feature
print("Range check (1-10):")
for j, nome in enumerate(FEATURES):
    col = extrair_coluna(X_train_bal, j)
    fora = [v for v in col if v < 1 or v > 10]
    ok = "✓" if len(fora) == 0 else f"✗ FORA! ({len(fora)} valores)"
    print(f"  {nome}: {ok}")

# Matriz de correlação (numpy puro)
X_train_bal_arr = np.array(X_train_bal)
corr = np.corrcoef(X_train_bal_arr, rowvar=False)

# Correlações altas (> 0.8)
high = []
for i in range(len(FEATURES)):
    for j in range(i + 1, len(FEATURES)):
        if abs(corr[i, j]) > 0.8:
            high.append((FEATURES[i], FEATURES[j], corr[i, j]))

print(f"\nCorrelações > 0.8: {len(high)}")
for f1, f2, v in high:
    print(f"  {f1} × {f2} = {v:.3f}")

# Heatmap com matplotlib puro (substitui sns.heatmap)
plt.figure(figsize=(10, 8))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(FEATURES)), FEATURES, rotation=45, ha='right', fontsize=8)
plt.yticks(range(len(FEATURES)), FEATURES, fontsize=8)
plt.title('Matriz de Correlação')

# Anotações nos quadrados
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        plt.text(j, i, f'{corr[i, j]:.2f}',
                 ha='center', va='center',
                 fontsize=7,
                 color='white' if abs(corr[i, j]) > 0.6 else 'black')

plt.tight_layout()
plt.savefig('correlacao_features.png', dpi=100, bbox_inches='tight')
plt.close()
print("\n  → correlacao_features.png salvo")


# ═══════════════════════════════════════════════════════════════
# 10. ITEM 13: Conversão de tipos + Normalização
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 13: Conversão de tipos + Normalização")
print("=" * 60)

# Converter Bare_nuclei float → int (após imputação, antes da normalização)
for linha in X_train_bal:
    linha[COL_BARE] = int(round(linha[COL_BARE]))
for linha in X_test_imp:
    linha[COL_BARE] = int(round(linha[COL_BARE]))
print("13a: Bare_nuclei float→int (após imputação)")
print("     Não necessário: nominal→binário, simbólico→numérico")

# Normalização Z-score: fit no treino, transform no treino e teste
X_train_scaled, medias_z, desvios_z = normalizar_zscore(X_train_bal)
X_test_scaled = aplicar_zscore(X_test_imp, medias_z, desvios_z)

# Verificar
X_train_scaled_arr = np.array(X_train_scaled)
print(f"13b: Z-score — média≈{X_train_scaled_arr.mean():.6f}, "
      f"std≈{X_train_scaled_arr.std():.6f}")


# ═══════════════════════════════════════════════════════════════
# 11. ITEM 14: PCA — Redução de dimensionalidade
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ITEM 14: PCA — Redução de dimensionalidade")
print("=" * 60)

# PCA no treino normalizado (calcula todos os autovalores, seleciona por limiar)
X_train_pca, componentes, var_explicada, medias_pca, n_comp = pca(
    X_train_scaled, n_componentes=None, limiar_variancia=0.95
)

# Variância acumulada
cum_var = np.cumsum(var_explicada)
for i, v in enumerate(cum_var):
    marcador = " ←" if i + 1 == n_comp else ""
    print(f"  {i + 1} comp: {v:.4f}{marcador}")

print(f"\nComponentes para 95%: {n_comp}")

# Gráfico PCA
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, 'bo-')
plt.axhline(0.95, color='r', linestyle='--', label='95%')
plt.axvline(n_comp, color='g', linestyle='--', label=f'n={n_comp}')
plt.xlabel('Componentes')
plt.ylabel('Variância Acumulada')
plt.legend()
plt.title('PCA — Variância Explicada')
plt.savefig('pca_variancia.png', dpi=100)
plt.close()
print("  → pca_variancia.png salvo")

# Aplicar PCA no teste
X_test_pca = aplicar_pca(X_test_scaled, componentes, medias_pca)

print(f"Dimensões PCA: treino={len(X_train_pca)}×{n_comp}, "
      f"teste={len(X_test_pca)}×{n_comp}")


# ═══════════════════════════════════════════════════════════════
# 12. SALVAR TUDO
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SALVANDO DADOS PRÉ-PROCESSADOS")
print("=" * 60)

# Arquivo .npz (arrays numpy)
np.savez('preprocessed_data.npz',
         X_train_scaled=np.array(X_train_scaled),
         X_test_scaled=np.array(X_test_scaled),
         X_train_pca=np.array(X_train_pca),
         X_test_pca=np.array(X_test_pca),
         y_train_bal=np.array(y_train_bal),
         y_test=np.array(y_test),
         feature_names=np.array(FEATURES),
         n_components_95=n_comp,
         explained_variance_ratio=np.array(var_explicada))
print("  → preprocessed_data.npz salvo")

# CSVs
salvar_csv('X_train_scaled.csv', X_train_scaled, cabecalho=FEATURES)
salvar_csv('X_test_scaled.csv', X_test_scaled, cabecalho=FEATURES)
salvar_csv('y_train_bal.csv', y_train_bal, cabecalho=['Class'])
salvar_csv('y_test.csv', y_test, cabecalho=['Class'])
print("  → CSVs salvos")

# Summary JSON
summary = {
    'dataset': 'Breast Cancer Wisconsin (UCI ID 15)',
    'total': total,
    'features': len(FEATURES),
    'feature_names': FEATURES,
    'target': 'Class (0=benigno, 1=maligno)',
    'train_balanced': len(y_train_bal),
    'test': len(y_test),
    'imputation': f'Bare_nuclei: mediana ({n_missing_train + n_missing_test} valores)',
    'balancing': 'SMOTE no treino',
    'scaling': 'StandardScaler (z-score)',
    'pca_components': n_comp,
    'pca_variance': float(cum_var[n_comp - 1]),
    'train_dist': {int(k): int(v) for k, v in contar_valores(y_train_bal).items()},
    'test_dist': {int(k): int(v) for k, v in contar_valores(y_test).items()}
}
with open('summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("  → summary.json salvo")

print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
print("\n✅ PRÉ-PROCESSAMENTO COMPLETO!")

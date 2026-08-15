"""
funcs_preprocessamento.py
Implementações de funções de pré-processamento.
"""

import math
import numpy as np
import csv


# ================================================================
# 1. LEITURA DE DADOS
# ================================================================

def carregar_csv(caminho):
    """Carrega CSV, retorna (dados_lista, cabecalho).
    dados_lista é lista de listas com floats. '?' vira math.nan."""
    with open(caminho, 'r') as f:
        leitor = csv.reader(f)
        cabecalho = next(leitor)
        dados = []
        for linha in leitor:
            linha_convertida = []
            for v in linha:
                v = v.strip()
                if v == '?' or v == '':
                    linha_convertida.append(float('nan'))
                else:
                    try:
                        linha_convertida.append(float(v))
                    except ValueError:
                        linha_convertida.append(v)
            dados.append(linha_convertida)
        return dados, cabecalho


def extrair_coluna(dados, indice):
    """Extrai uma coluna como lista de valores (ignora None/NaN)."""
    return [linha[indice] for linha in dados if linha[indice] is not None 
            and not (isinstance(linha[indice], float) and math.isnan(linha[indice]))]


def extrair_matriz(dados, indices_colunas):
    """Extrai submatriz com as colunas especificadas. Retorna lista de listas."""
    return [[linha[i] for i in indices_colunas] for linha in dados]


def substituir_coluna(matriz, indice, novos_valores):
    """Substitui uma coluna da matriz por novos valores."""
    for i, linha in enumerate(matriz):
        linha[indice] = novos_valores[i]
    return matriz


# ================================================================
# 2. ESTATÍSTICAS DESCRITIVAS
# ================================================================

def media(dados):
    """Média aritmética."""
    d = [x for x in dados if x is not None]
    if not d:
        return 0.0
    return sum(d) / len(d)


def mediana(dados):
    """Mediana."""
    d = sorted([x for x in dados if x is not None])
    if not d:
        return 0.0
    n = len(d)
    if n % 2 == 1:
        return d[n // 2]
    return (d[n // 2 - 1] + d[n // 2]) / 2.0


def moda(dados):
    """Moda (primeira se múltiplas)."""
    from collections import Counter
    d = [x for x in dados if x is not None]
    if not d:
        return None
    return Counter(d).most_common(1)[0][0]


def desvio_padrao(dados, amostral=True):
    """Desvio padrão."""
    d = [x for x in dados if x is not None]
    n = len(d)
    if n < 2:
        return 0.0
    m = media(d)
    var = sum((x - m) ** 2 for x in d) / (n - 1 if amostral else n)
    return math.sqrt(var)


def variancia(dados, amostral=True):
    """Variância."""
    d = [x for x in dados if x is not None]
    n = len(d)
    if n < 2:
        return 0.0
    m = media(d)
    return sum((x - m) ** 2 for x in d) / (n - 1 if amostral else n)


def percentil(dados, p):
    """Percentil p (0-100) usando interpolação linear."""
    d = sorted([x for x in dados if x is not None])
    if not d:
        return 0.0
    n = len(d)
    k = (p / 100.0) * (n - 1)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return d[f]
    return d[f] + (k - f) * (d[c] - d[f])


def iqr(dados):
    """Intervalo interquartil."""
    return percentil(dados, 75) - percentil(dados, 25)


def minimo(dados):
    """Valor mínimo."""
    d = [x for x in dados if x is not None]
    return min(d) if d else None


def maximo(dados):
    """Valor máximo."""
    d = [x for x in dados if x is not None]
    return max(d) if d else None


def amplitude(dados):
    """Amplitude (max - min)."""
    d = [x for x in dados if x is not None]
    return max(d) - min(d) if d else 0


def skewness(dados):
    """Coeficiente de assimetria (Fisher-Pearson)."""
    d = [x for x in dados if x is not None]
    n = len(d)
    if n < 3:
        return 0.0
    m = media(d)
    s = desvio_padrao(d, amostral=True)
    if s == 0:
        return 0.0
    soma = sum(((x - m) / s) ** 3 for x in d)
    return (n / ((n - 1) * (n - 2))) * soma


def kurtosis(dados):
    """Curtose (excesso, Fisher)."""
    d = [x for x in dados if x is not None]
    n = len(d)
    if n < 4:
        return 0.0
    m = media(d)
    s = desvio_padrao(d, amostral=True)
    if s == 0:
        return 0.0
    soma = sum(((x - m) / s) ** 4 for x in d)
    termo1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    termo2 = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return termo1 * soma - termo2


# ================================================================
# 3. MANIPULAÇÃO DE DADOS
# ================================================================

def train_test_split_estratificado(X, y, test_size=0.2, seed=42):
    """Split estratificado manual: mantém proporção das classes.
    X: lista de listas (features)
    y: lista (targets)
    Retorna: X_train, X_test, y_train, y_test (listas de listas)"""
    import random
    random.seed(seed)
    
    # Agrupar índices por classe
    classes = {}
    for i, classe in enumerate(y):
        classes.setdefault(classe, []).append(i)
    
    train_idx, test_idx = [], []
    for classe, indices in classes.items():
        random.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_idx.extend(indices[:n_test])
        train_idx.extend(indices[n_test:])
    
    random.shuffle(train_idx)
    random.shuffle(test_idx)
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    
    return X_train, X_test, y_train, y_test


def detectar_valores_ausentes(matriz):
    """Retorna lista de (linha, coluna) com valores None ou NaN."""
    ausentes = []
    for i, linha in enumerate(matriz):
        for j, val in enumerate(linha):
            if val is None or (isinstance(val, float) and math.isnan(val)):
                ausentes.append((i, j))
    return ausentes


def imputar_mediana(matriz):
    """Imputa valores ausentes com a mediana da coluna."""
    if not matriz:
        return matriz
    n_cols = len(matriz[0])
    resultado = [list(linha) for linha in matriz]
    
    for j in range(n_cols):
        coluna = extrair_coluna(matriz, j)
        med = mediana(coluna)
        for i in range(len(matriz)):
            val = resultado[i][j]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                resultado[i][j] = med
    return resultado


def contar_duplicatas(matriz):
    """Conta linhas duplicadas (ignora None para comparação)."""
    vistas = set()
    duplicatas = 0
    for linha in matriz:
        t = tuple(linha)
        if t in vistas:
            duplicatas += 1
        else:
            vistas.add(t)
    return duplicatas


# ================================================================
# 4. NORMALIZAÇÃO
# ================================================================

def normalizar_zscore(matriz):
    """StandardScaler manual: z = (x - μ) / σ.
    Retorna: (matriz_normalizada, medias, desvios)"""
    X = np.array(matriz, dtype=np.float64)
    medias = np.mean(X, axis=0)
    desvios = np.std(X, axis=0, ddof=1)
    desvios[desvios == 0] = 1.0  # evitar divisão por zero
    X_norm = (X - medias) / desvios
    return X_norm.tolist(), medias.tolist(), desvios.tolist()


def aplicar_zscore(matriz, medias, desvios):
    """Aplica z-score com médias e desvios pré-calculados."""
    X = np.array(matriz, dtype=np.float64)
    return ((X - np.array(medias)) / np.array(desvios)).tolist()


# ================================================================
# 5. SMOTE (Synthetic Minority Over-sampling Technique)
# ================================================================

def smote(X, y, k=5, seed=42):
    """SMOTE manual: gera exemplos sintéticos da classe minoritária.
    X: lista de listas (features)
    y: lista (targets binários 0/1)
    k: número de vizinhos
    Retorna: (X_balanced, y_balanced)"""
    import random
    random.seed(seed)
    
    X_arr = np.array(X, dtype=np.float64)
    y_arr = np.array(y)
    
    # Encontrar classe minoritária
    classes, contagens = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return X, y
    
    idx_majoritaria = np.argmax(contagens)
    idx_minoritaria = 1 - idx_majoritaria
    classe_minoritaria = classes[idx_minoritaria]
    
    # Índices da classe minoritária
    indices_min = np.where(y_arr == classe_minoritaria)[0]
    X_min = X_arr[indices_min]
    
    # Quantos sintéticos gerar
    n_majoritaria = contagens[idx_majoritaria]
    n_minoritaria = contagens[idx_minoritaria]
    n_sinteticos = n_majoritaria - n_minoritaria
    
    if n_sinteticos <= 0:
        return X, y  # já balanceado
    
    # Função de distância euclidiana
    def distancia_euclidiana(a, b):
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
    
    # Gerar exemplos sintéticos
    sinteticos = []
    for _ in range(n_sinteticos):
        # Escolher exemplo base aleatório
        idx_base = random.randint(0, len(X_min) - 1)
        x_base = X_min[idx_base]
        
        # Encontrar k vizinhos mais próximos
        distancias = []
        for j, x_j in enumerate(X_min):
            if j != idx_base:
                distancias.append((distancia_euclidiana(x_base, x_j), j))
        distancias.sort()
        
        # Escolher um vizinho aleatório entre os k mais próximos
        k_real = min(k, len(distancias))
        idx_vizinho = distancias[random.randint(0, k_real - 1)][1]
        x_vizinho = X_min[idx_vizinho]
        
        # Interpolar
        gap = random.random()
        x_sintetico = x_base + gap * (x_vizinho - x_base)
        sinteticos.append(x_sintetico.tolist())
    
    # Combinar dados originais com sintéticos
    X_balanced = X + sinteticos
    y_balanced = y + [classe_minoritaria] * len(sinteticos)
    
    # Embaralhar
    combinado = list(zip(X_balanced, y_balanced))
    random.shuffle(combinado)
    X_balanced = [c[0] for c in combinado]
    y_balanced = [c[1] for c in combinado]
    
    return X_balanced, y_balanced


# ================================================================
# 6. PCA (Principal Component Analysis)
# ================================================================

def pca(matriz, n_componentes=None, limiar_variancia=0.95):
    """PCA manual usando numpy.linalg.eig.
    matriz: lista de listas (já normalizada)
    n_componentes: número de componentes (se None, usa limiar)
    limiar_variancia: variância acumulada mínima
    Retorna: (matriz_transformada, componentes, variancia_explicada, 
              medias, n_componentes_usados)"""
    X = np.array(matriz, dtype=np.float64)
    
    # Centralizar (já deve estar normalizada, mas garantimos)
    medias = np.mean(X, axis=0)
    X_centered = X - medias
    
    # Matriz de covariância
    cov = np.cov(X_centered, rowvar=False)
    
    # Autovalores e autovetores
    autovalores, autovetores = np.linalg.eig(cov)
    
    # Ordenar por autovalor decrescente
    idx = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx]
    autovetores = autovetores[:, idx]
    
    # Variância explicada
    var_total = np.sum(autovalores)
    var_explicada = autovalores / var_total
    var_acumulada = np.cumsum(var_explicada)
    
    # Determinar n_componentes
    if n_componentes is None:
        n_componentes = int(np.searchsorted(var_acumulada, limiar_variancia) + 1)
    
    # Selecionar componentes
    componentes = autovetores[:, :n_componentes]
    
    # Transformar
    X_transformado = X_centered @ componentes
    
    return (X_transformado.tolist(), componentes.tolist(), 
            var_explicada.tolist(), medias.tolist(), n_componentes)


def aplicar_pca(matriz, componentes, medias):
    """Aplica PCA com componentes e médias pré-calculados."""
    X = np.array(matriz, dtype=np.float64)
    X_centered = X - np.array(medias)
    return (X_centered @ np.array(componentes)).tolist()


# ================================================================
# 7. OUTLIERS
# ================================================================

def detectar_outliers_iqr(matriz):
    """Detecta outliers via IQR. Retorna máscara booleana por coluna."""
    X = np.array(matriz, dtype=np.float64)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr_val = q3 - q1
    limite_inf = q1 - 1.5 * iqr_val
    limite_sup = q3 + 1.5 * iqr_val
    
    outliers = (X < limite_inf) | (X > limite_sup)
    return outliers.sum(axis=0).tolist()


# ================================================================
# 8. UTILITÁRIOS
# ================================================================

def valores_unicos(lista):
    """Valores únicos ordenados."""
    return sorted(set(x for x in lista if x is not None))


def contar_valores(lista):
    """Contagem de cada valor distinto."""
    contagem = {}
    for x in lista:
        if x is not None:
            contagem[x] = contagem.get(x, 0) + 1
    return contagem


def proporcoes(lista):
    """Proporção de cada valor."""
    contagem = contar_valores(lista)
    total = sum(contagem.values())
    return {k: v / total for k, v in contagem.items()}

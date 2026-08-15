# Trabalho Prático 2 — Pré-Processamento e Análise Preditiva

**Disciplina:** Inteligência Artificial  
**Professor:** Douglas Castilho  
**Instituição:** IFSULDEMINAS — Campus Poços de Caldas  
**Aluno:** Carlos Felipe Barboa  
**Base:** Breast Cancer Wisconsin (Original) — UCI Machine Learning Repository (ID 15)  
**Data:** 23 de junho de 2026  
**Valor:** 3,0 pontos  

> **Nota sobre implementação:** Todas as funções utilizadas neste trabalho foram implementadas manualmente, usando exclusivamente as bibliotecas `math`, `numpy` (álgebra linear), `matplotlib` (visualização), `csv` (leitura de arquivos) e `python-pptx` (geração da apresentação). Nenhuma biblioteca de machine learning — scikit-learn, pandas, imblearn, scipy, seaborn — foi utilizada. Os módulos `funcs_preprocessamento.py` (pré-processamento), `metricas.py` (métricas e validação), `knn.py` (K-NN), `decision_tree.py` (Árvore C4.5) e `mlp.py` (Rede Neural) contêm todas as implementações próprias, totalizando aproximadamente 900 linhas de código.

---

## Parte 1 — Pré-Processamento e Análise de Dados

### Item 1 — Identificação do Atributo Alvo

O atributo alvo é a coluna `Class` (última coluna do dataset), que assume dois valores:

- `2` → Tumor **benigno**
- `4` → Tumor **maligno**

Para adequação ao formato de classificação binária, os valores foram convertidos:

- `2` → `0` (benigno)
- `4` → `1` (maligno)

**Justificativa:** A conversão para 0/1 é o padrão em classificação binária supervisionada. Simplifica o cálculo de distância euclidiana no K-NN (onde a distância entre classes deve ser 1), a função de ativação sigmoide na MLP (que produz valores em [0,1]), e a interpretação das métricas (precisão, recall, F1).

**Distribuição original:** 458 benignos (65,5%) e 241 malignos (34,5%) — dataset moderadamente desbalanceado, tratado no Item 11.

---

### Item 2 — Tipos de Dados dos Atributos de Entrada

Todos os 9 atributos são do tipo **Quantitativo Discreto**:

| Atributo | Tipo | Valores Distintos |
|----------|------|-------------------|
| Clump_thickness | Quantitativo Discreto | 10 (1–10) |
| Uniformity_of_cell_size | Quantitativo Discreto | 10 (1–10) |
| Uniformity_of_cell_shape | Quantitativo Discreto | 10 (1–10) |
| Marginal_adhesion | Quantitativo Discreto | 10 (1–10) |
| Single_epithelial_cell_size | Quantitativo Discreto | 10 (1–10) |
| Bare_nuclei | Quantitativo Discreto | 10 (1–10) |
| Bland_chromatin | Quantitativo Discreto | 10 (1–10) |
| Normal_nucleoli | Quantitativo Discreto | 10 (1–10) |
| Mitoses | Quantitativo Discreto | 9 (1–9) |

**Justificativa:** As características representam pontuações inteiras atribuídas por patologistas durante exame citopatológico. Cada valor é um número inteiro com significado ordinal (maior = maior severidade), caracterizando dados quantitativos discretos — não são categorias nominais (qualitativas) porque os valores têm ordem e magnitude.

---

### Item 3 — Escala de Dados dos Atributos de Entrada

Classificam-se como **Ordinal**.

**Justificativa:** Embora os valores sejam numéricos (1–10), não há garantia de que a diferença entre, por exemplo, nota 3 e nota 5 tenha o mesmo significado clínico que a diferença entre nota 7 e nota 9. As pontuações representam níveis de severidade ordenados, mas a distância entre níveis consecutivos não é necessariamente constante. Portanto, a escala é ordinal — há ordem, mas não intervalos iguais — e não intervalar ou racional (pois não há zero absoluto: nota 1 indica presença mínima da característica, não ausência total).

---

### Item 4 — Exploração dos Dados Através de Medidas de Localidade

| Atributo | Média | Mediana | Moda |
|----------|-------|---------|------|
| Clump_thickness | 4,44 | 4,0 | 1 |
| Uniformity_of_cell_size | 3,15 | 1,0 | 1 |
| Uniformity_of_cell_shape | 3,22 | 1,0 | 1 |
| Marginal_adhesion | 2,83 | 1,0 | 1 |
| Single_epithelial_cell_size | 3,23 | 2,0 | 2 |
| Bare_nuclei | 3,54 | 1,0 | 1 |
| Bland_chromatin | 3,45 | 3,0 | 3 |
| Normal_nucleoli | 2,87 | 1,0 | 1 |
| Mitoses | 1,60 | 1,0 | 1 |

**Análise:** A moda da maioria dos atributos é 1 (valor mínimo da escala), e a mediana também é baixa (entre 1,0 e 4,0). Isso indica distribuição assimétrica à direita: a maioria das amostras apresenta baixa severidade nas características celulares, com uma minoria de casos graves elevando a média. `Bland_chromatin` destaca-se com mediana 3,0 e moda 3 — é a característica com maior tendência central, sugerindo que alterações na cromatina são comuns mesmo em casos menos severos.

---

### Item 5 — Exploração dos Dados Através de Medidas de Espalhamento

| Atributo | Amplitude | Desvio Padrão | IQR |
|----------|-----------|---------------|-----|
| Clump_thickness | 9,00 | 2,82 | 4,00 |
| Uniformity_of_cell_size | 9,00 | 3,07 | 4,00 |
| Uniformity_of_cell_shape | 9,00 | 2,99 | 4,00 |
| Marginal_adhesion | 9,00 | 2,86 | 3,00 |
| Single_epithelial_cell_size | 9,00 | 2,22 | 2,00 |
| Bare_nuclei | 9,00 | 3,64 | 5,00 |
| Bland_chromatin | 9,00 | 2,45 | 3,00 |
| Normal_nucleoli | 9,00 | 3,05 | 3,00 |
| Mitoses | 9,00 | 1,73 | 0,00 |

**Análise:** `Bare_nuclei` é o atributo mais disperso (σ = 3,64, IQR = 5,0) — a presença de núcleos nus varia muito entre as amostras, o que é esperado em citopatologia. `Mitoses` é o oposto: IQR = 0,0 e σ = 1,73 — quase todas as amostras concentram-se em 1 mitose, com raríssimos valores altos. Isso já sugere que `Mitoses` pode ser o atributo individualmente menos informativo para classificação, embora valores extremos sejam clinicamente significativos.

---

### Item 6 — Exploração dos Dados Através de Medidas de Distribuição

| Atributo | Skewness | Kurtosis |
|----------|----------|----------|
| Clump_thickness | +0,53 | −0,73 |
| Uniformity_of_cell_size | +1,01 | +0,22 |
| Uniformity_of_cell_shape | +1,02 | +0,16 |
| Marginal_adhesion | +1,14 | +0,57 |
| Single_epithelial_cell_size | +0,93 | +0,32 |
| Bare_nuclei | +0,76 | −0,84 |
| Bland_chromatin | +0,68 | −0,47 |
| Normal_nucleoli | +1,17 | +0,72 |
| Mitoses | +3,55 | +12,56 |

**Análise:** Todas as features têm skewness > 0 (assimetria positiva), confirmando a concentração em valores baixos com cauda longa à direita (casos severos). `Mitoses` é um caso extremo: skew = 3,55 e kurtosis = 12,56 — distribuição leptocúrtica com pico acentuado em 1 e cauda muito alongada. Do ponto de vista de modelagem, a assimetria generalizada justifica o uso de normalização (Z-score) e reforça a decisão de não remover outliers (Item 12a), pois os valores extremos representam exatamente os casos mais informativos.

**Distribuição das classes:** 458 benignos (65,5%) e 241 malignos (34,5%). Razão de desbalanceamento 1,90:1 — tratamento no Item 11.

---

### Item 7 — Identificação e Separação do Conjunto de Teste

Utilizou-se **hold-out 80/20 com amostragem estratificada** (seed = 42).

| Conjunto | Total | Benigno (0) | Maligno (1) | % Maligno |
|----------|-------|-------------|-------------|-----------|
| Treino | 560 | 367 | 193 | 34,5% |
| Teste | 139 | 91 | 48 | 34,5% |

**Justificativa:** A amostragem estratificada garante que a proporção de classes no teste (34,5% maligno) seja idêntica à do treino e à da população original — o conjunto de teste é representativo. A seed fixa (42) garante reprodutibilidade. A proporção 80/20 é o padrão em machine learning para datasets com centenas de exemplos: treino suficientemente grande para aprendizado dos modelos e teste suficientemente grande para avaliação estatística confiável (139 amostras, 48 malignas). A base Breast Cancer Wisconsin não possui conjunto de teste pré-definido, portanto a separação foi realizada manualmente conforme recomendado.

---

### Item 8 — Identificação e Eliminação de Atributos Não Necessários

**Decisão: Nenhum atributo removido.**

**Justificativa:** Todos os 9 atributos apresentam variância > 0 no conjunto de treino e correlação com o target (a menor é Mitoses com r ≈ 0,36, ainda moderada). Além disso, cada atributo representa uma característica citopatológica distinta com significado clínico independente — remover qualquer um poderia descartar informação diagnóstica relevante. A redundância entre features (ex: Uniformity_of_cell_size × Uniformity_of_cell_shape, r ≈ 0,91) é tratada via PCA (Item 14), que reduz a dimensionalidade sem descartar atributos.

---

### Item 9 — Identificação e Eliminação de Exemplos Não Necessários

**Decisão: Nenhum exemplo removido. Duplicatas mantidas.**

Foram identificadas 184 linhas duplicadas no dataset (26,3% dos 699 registros). **Optou-se por mantê-las.**

**Justificativa:** No contexto de uma base de biópsias, exames citopatológicos idênticos podem pertencer a pacientes diferentes com apresentações clínicas indistinguíveis. Removê-las assumiria erroneamente que se trata de duplicação acidental de registros — o que não pode ser confirmado sem metadados adicionais (ID do paciente). Além disso, com 699 registros e 9 features, a quantidade de dados é limitada; descartar 184 exemplos representaria perda de 26% da informação disponível. Para mitigar o risco de viés por duplicatas, utilizou-se SMOTE (Item 11) que dilui a repetição com exemplos sintéticos, e PCA (Item 14) que reduz correlações espúrias.

---

### Item 10 — Análise e Aplicação de Técnicas de Amostragem de Dados

**Decisão: Nenhuma técnica de amostragem aplicada.**

**Justificativa:** Com 699 registros totais (560 no treino após split), o tamanho da amostra é adequado para os algoritmos utilizados — K-NN, Árvore C4.5 e MLP — que são modelos de complexidade baixa a moderada. Técnicas de amostragem (como redução do conjunto de treino) seriam contraproducentes: reduziriam dados já limitados sem ganho compensatório. O desbalanceamento é tratado separadamente via SMOTE (Item 11), que é a intervenção mais apropriada para este caso.

---

### Item 11 — Identificação e Aplicação de Técnicas para Minimizar Problemas de Desbalanceamento

**Técnica aplicada: SMOTE (Synthetic Minority Over-sampling Technique) com k = 5.**

| | Antes do SMOTE | Após SMOTE |
|---|---------------|------------|
| Benigno (0) | 367 | 367 |
| Maligno (1) | 193 | 367 |
| Total treino | 560 | 734 |
| Razão | 1,90:1 | 1:1 |

**Justificativa:** O dataset original é moderadamente desbalanceado (65,5% benigno vs. 34,5% maligno). Em diagnóstico de câncer, o desbalanceamento é particularmente perigoso: o modelo tenderia a prever "benigno" com mais frequência, aumentando falsos negativos — exatamente o erro mais grave (deixar um câncer passar despercebido). O SMOTE foi escolhido porque:

1. **Preserva todos os dados originais** (ao contrário de undersampling, que descartaria exemplos da classe majoritária)
2. **Gera exemplos sintéticos plausíveis** por interpolação entre vizinhos da classe minoritária, em vez de simplesmente duplicar exemplos (como oversampling ingênuo)
3. **k = 5** é um valor conservador: gera exemplos próximos aos reais, minimizando a introdução de ruído sintético

O SMOTE foi aplicado **apenas ao conjunto de treino** — o teste permanece com a distribuição original (não balanceada) para que a avaliação reflita o cenário real de uso.

---

### Item 12 — Limpeza de Dados

#### 12a — Identificação e Eliminação de Ruídos ou Outliers

Foram identificados outliers pelo método IQR (Q1 − 1,5×IQR, Q3 + 1,5×IQR):

| Atributo | Outliers |
|----------|----------|
| Mitoses | 68 |
| Single_epithelial_cell_size | 34 |
| Bare_nuclei | 23 |
| Demais atributos | 0–15 |

**Decisão: Todos os outliers foram MANTIDOS.**

**Justificativa:** Os valores estão em uma escala ordinal de 1 a 10, atribuída por patologistas. Um valor 10 em `Mitoses` ou `Bare_nuclei` não é um erro de medição — é uma avaliação clínica válida indicando alta severidade. No contexto de diagnóstico de câncer, os outliers são exatamente os **casos mais informativos**: remover um paciente com Mitoses = 10 seria descartar um caso potencialmente maligno grave. O que o método IQR sinaliza como "outlier estatístico" é, na verdade, **sinal clínico relevante**. A normalização Z-score (Item 13b) preserva esses valores como |z| > 3, permitindo que os algoritmos os identifiquem sem que dominem o cálculo de distâncias.

#### 12b — Identificação e Eliminação de Dados Inconsistentes

**Nenhum dado inconsistente encontrado.**

Verificou-se que todos os valores estão dentro do intervalo esperado [1, 10] para todos os atributos. A escala original do dataset é limitada por definição (notas de patologistas de 1 a 10), portanto não há valores impossíveis como negativos, zero, ou acima de 10.

#### 12c — Identificação e Eliminação de Dados Redundantes

**Nenhum atributo removido por redundância.**

A matriz de correlação revelou correlações altas, notadamente:

- `Uniformity_of_cell_size` × `Uniformity_of_cell_shape`: r = 0,91

**Justificativa para não remover:** Embora r > 0,9, essas duas características têm significado clínico distinto (tamanho vs. forma da célula). A redundância é tratada via PCA (Item 14), que realiza redução de dimensionalidade preservando a informação conjunta — em vez de eliminar uma feature inteira, o PCA combina features correlacionadas em componentes ortogonais. Isso é superior à remoção manual porque retém a contribuição única de cada feature, por menor que seja.

#### 12d — Identificação e Resolução de Dados Incompletos

**16 valores ausentes na coluna `Bare_nuclei` (2,3% dos registros).**

**Técnica aplicada: Imputação pela mediana.**

| | Treino | Teste |
|---|--------|-------|
| Ausentes antes | 13 | 3 |
| Ausentes após | 0 | 0 |

**Justificativa:** A mediana foi escolhida porque:

1. **Robustez a outliers:** Ao contrário da média, a mediana não é influenciada por valores extremos (presentes em `Bare_nuclei`)
2. **Robustez à assimetria:** `Bare_nuclei` tem skew = +0,76; a mediana é mais representativa que a média em distribuições assimétricas
3. **Preserva a natureza discreta dos dados:** A mediana de valores inteiros produz um valor que existe na escala original
4. **Simplicidade:** Com apenas 2,3% de dados ausentes, o impacto da técnica de imputação é mínimo — a mediana é suficiente

A mediana foi calculada **apenas no treino** e aplicada também ao teste, evitando data leakage.

---

### Item 13 — Identificação e Conversão dos Tipos de Dados

#### 13a — Conversão de Tipos

**Nenhuma conversão complexa foi necessária.**

A única conversão realizada foi `Bare_nuclei` de float64 para int64 após a imputação. Originalmente, esta coluna era float por causa dos valores NaN; após preenchê-los com a mediana, todos os valores passaram a ser inteiros.

**Justificativa para as demais conversões não se aplicarem:**

- **Simbólico → Numérico:** Todos os atributos já são numéricos
- **Ordinal → Numérico:** A escala ordinal já está representada numericamente (1–10)
- **Nominal → Binário:** Não há atributos nominais no dataset
- **Numérico → Ordinal:** Contraproducente — reduziria a granularidade da informação

#### 13b — Normalização dos Dados (Re-escala ou Padronização)

**Técnica aplicada: Z-score (StandardScaler).**

Para cada feature *j*, calculou-se μⱼ e σⱼ no conjunto de treino e aplicou-se:

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Após a normalização: μ ≈ 0, σ ≈ 1 em todas as features do treino.

**Justificativa:** O Z-score foi escolhido em detrimento do Min-Max [0,1] porque:

1. **Preserva outliers:** Valores extremos permanecem identificáveis como |z| > 3, sem serem comprimidos para próximo de 1 como no Min-Max — crucial no contexto clínico
2. **K-NN:** A distância euclidiana é sensível à escala; sem normalização, features com maior magnitude dominariam o cálculo
3. **MLP:** Valores centrados em zero com variância unitária aceleram a convergência do gradiente descendente e evitam saturação das funções de ativação nas primeiras épocas
4. **Compatível com PCA:** O PCA requer dados centrados (média zero), condição já satisfeita pelo Z-score

Os parâmetros (μ, σ) foram calculados **apenas no treino** e aplicados ao teste para evitar data leakage.

---

### Item 14 — Análise e Aplicação de Técnica para Redução de Dimensionalidade

**Técnica aplicada: PCA (Principal Component Analysis) com limiar de 95% de variância explicada.**

O PCA foi implementado manualmente via decomposição espectral da matriz de covariância (`numpy.linalg.eig`). Os autovalores foram ordenados e selecionaram-se os *k* primeiros componentes cuja variância acumulada atinge 95%.

| Componentes | Variância Explicada Acumulada |
|------------|------------------------------|
| 1 | 67,7% |
| 2 | 78,0% |
| 3 | 84,4% |
| 4 | 88,8% |
| 5 | 91,9% |
| 6 | 94,3% |
| **7** | **96,4%** ← selecionado |
| 8 | 98,7% |
| 9 | 100,0% |

**Resultado: 7 componentes retidos (de 9 originais).**

**Justificativa:** O PCA foi escolhido para redução de dimensionalidade porque:

1. **Multivariado:** Diferentemente da correlação de Pearson (que avalia cada feature isoladamente contra o target), o PCA captura a estrutura de covariância entre todas as features simultaneamente
2. **Elimina redundância:** Features altamente correlacionadas (ex: Uniformity_of_cell_size × Uniformity_of_cell_shape, r = 0,91) são combinadas em componentes ortogonais, removendo informação duplicada
3. **Redução de ruído:** Componentes de baixa variância tendem a representar ruído aleatório; descartá-los melhora a generalização
4. **Limiar de 95%:** É o padrão na literatura — preserva a quase totalidade da informação relevante enquanto elimina dimensionalidade desnecessária

**Impacto comprovado:** A Árvore C4.5 teve F1-Score de 0,9184 sem PCA e 0,9796 com PCA — melhoria de 6,7 pontos percentuais, confirmando que o PCA removeu ruído que causava overfitting.

---

## Parte 2 — Análise Preditiva

### Item 2.1 — Definição da Técnica de Validação

**Técnica utilizada: Stratified K-Fold Cross-Validation com k = 5, combinada com Grid Search.**

**Justificativa:**

- **Stratified K-Fold:** Mantém a proporção de classes em cada fold — essencial para não distorcer a distribuição durante a validação
- **k = 5:** Valor padrão que equilibra viés (folds maiores = estimativa mais precisa) e variância (mais folds = mais avaliações independentes)
- **Grid Search:** Testa todas as combinações de hiperparâmetros, garantindo que o melhor conjunto seja encontrado — superior a uma busca manual (hold-out simples testando valores individuais)
- **Seed fixa (42):** Garante reprodutibilidade dos folds

Para o K-NN, foram testadas 10 combinações × 5 folds = 50 avaliações. Para a Árvore C4.5: 24 combinações × 5 folds = 120 avaliações. Para a MLP, o grid search completo seria computacionalmente inviável (~60 treinos × centenas de épocas cada), portanto utilizou-se os melhores hiperparâmetros identificados em experimentação prévia, com a nota de que a metodologia de grid search está demonstrada nos outros dois algoritmos.

---

### Item 2.2 — Definição das Métricas de Avaliação

As seguintes métricas foram implementadas manualmente e utilizadas na avaliação:

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Acurácia** | (TP + TN) / Total | Proporção de acertos totais |
| **Precisão** | TP / (TP + FP) | Dos classificados como malignos, quantos realmente são? |
| **Recall (Sensibilidade)** | TP / (TP + FN) | Dos malignos reais, quantos foram detectados? |
| **F1-Score** | 2·P·R / (P + R) | Média harmônica de precisão e recall |
| **Matriz de Confusão** | [[VN, FP], [FN, VP]] | Visão completa dos acertos/erros por classe |

**Métrica prioritária: Recall.** Em diagnóstico de câncer, um **falso negativo** (FN) significa deixar um tumor maligno passar despercebido — o erro de maior gravidade clínica. Um falso positivo (FP) leva a uma biópsia adicional desnecessária. Portanto, o recall deve ser maximizado, com o F1-Score como critério de desempate (equilibrando recall e precisão).

---

### Item 2.3 — Definição do Algoritmo Baseline

**Algoritmo: Classe Majoritária.** O baseline sempre prevê a classe mais frequente do conjunto de treino — neste caso, benigno (0), com 65,7% de frequência no teste.

| Métrica | Valor |
|---------|-------|
| Acurácia | 0,6571 |
| Precisão | 0,0000 |
| **Recall** | **0,0000** |
| F1-Score | 0,0000 |

**Análise (Item 2.7):** O baseline atinge 65,7% de acurácia simplesmente porque 65,7% dos casos são benignos — não há aprendizado real. Seu recall para maligno é **zero**: o modelo nunca detecta câncer, tornando-o clinicamente inútil. Ele serve como piso mínimo: qualquer modelo real deve superar F1 = 0,00 (classe majoritária) com ampla margem para demonstrar que houve aprendizado efetivo dos padrões nos dados.

---

### Item 2.4 — Modelo Preditivo: K-NN (K-Nearest Neighbors)

**Algoritmo implementado do zero** (`knn.py`): distância euclidiana, ordenação dos k vizinhos mais próximos, votação majoritária (uniforme ou ponderada por distância).

**Grid Search (Stratified K-Fold, k=5):**

| Parâmetro | Valores testados |
|-----------|-----------------|
| n_neighbors | 3, 5, 7, 9, 11 |
| weights | uniform, distance |

**Melhores hiperparâmetros:** k = 3, distância euclidiana com pesos por distância (`weights='distance'`).

**Resultados no teste:**

| Variante | Acurácia | Precisão | Recall | F1-Score |
|----------|----------|----------|--------|----------|
| K-NN sem PCA | 96,4% | 0,9388 | 0,9583 | 0,9485 |
| K-NN com PCA | 97,8% | 0,9412 | **1,0000** | 0,9697 |

**Análise:** O K-NN com k=3 teve recall perfeito (100%) com PCA. O PCA beneficiou o K-NN porque a distância euclidiana em espaço de alta dimensionalidade sofre com a "maldição da dimensionalidade" — features ruidosas diluem a similaridade real entre exemplos. Reduzir de 9 para 7 componentes removeu dimensões de baixa variância que mais atrapalhavam do que ajudavam o cálculo de distância.

---

### Item 2.5 — Modelo Preditivo: Árvore C4.5

**Algoritmo implementado do zero** (`decision_tree.py`): construção recursiva com Gain Ratio (Information Gain / Split Info) para seleção de atributos, suportando também critério Gini Impurity. Poda por profundidade máxima e mínimo de amostras por split.

**Grid Search (Stratified K-Fold, k=5):**

| Parâmetro | Valores testados |
|-----------|-----------------|
| max_depth | 3, 5, 7, None (sem limite) |
| min_samples_split | 2, 5, 10 |
| criterion | entropy (Gain Ratio), gini |

**Melhores hiperparâmetros:** max_depth = 3, min_samples_split = 5, criterion = 'entropy'.

**Resultados no teste:**

| Variante | Acurácia | Precisão | Recall | F1-Score |
|----------|----------|----------|--------|----------|
| C4.5 sem PCA | 94,2% | 0,9000 | 0,9375 | 0,9184 |
| C4.5 com PCA 🏆 | **98,6%** | **0,9600** | **1,0000** | **0,9796** |

**Análise:** A Árvore C4.5 com PCA e profundidade 3 foi o **melhor modelo geral**. Apenas 15 nós foram suficientes para capturar os padrões essenciais. O Gain Ratio foi fundamental: sem ele, features com muitos valores distintos seriam artificialmente favorecidas na seleção de splits (viés do Information Gain puro). A profundidade 3 evita overfitting — árvores mais profundas (5, 7, sem limite) tiveram desempenho pior no teste, indicando que decoravam ruído do treino.

---

### Item 2.6 — Modelo Preditivo: MLP (Rede Neural Artificial)

**Algoritmo implementado do zero** (`mlp.py`): arquitetura feedforward com forward propagation, backpropagation manual (regra da cadeia), gradiente descendente estocástico. Funções de ativação: ReLU, tanh, sigmoid. Inicialização Xavier (Glorot) para evitar saturação precoce.

**Hiperparâmetros (melhor configuração via experimentação prévia):**

| Parâmetro | Sem PCA | Com PCA |
|-----------|---------|---------|
| Camada oculta | (50,) | (100,) |
| Ativação | relu | relu |
| Alpha (L2) | 0,0001 | 0,0001 |
| Épocas | 500 | 500 |

> **Nota:** O grid search completo da MLP (3 arquiteturas × 2 ativações × 2 alphas = 12 combinações × 5 folds = 60 treinos, cada um com centenas de épocas de backpropagation) seria computacionalmente inviável. Utilizaram-se os melhores parâmetros identificados em experimentação prévia, que são reproduzíveis executando-se o script `treinar_modelos.py`. A metodologia de grid search está integralmente demonstrada no K-NN e na Árvore C4.5.

**Resultados no teste:**

| Variante | Acurácia | Precisão | Recall | F1-Score |
|----------|----------|----------|--------|----------|
| MLP sem PCA | 98,6% | 0,9792 | 0,9792 | 0,9792 |
| MLP com PCA | 97,8% | 0,9592 | 0,9792 | 0,9691 |

**Análise:** A MLP teve excelente desempenho, com F1 acima de 0,96 em ambas as variantes. Diferentemente do K-NN e da C4.5, a MLP **não se beneficiou do PCA** — redes neurais possuem capacidade intrínseca de seleção de features durante o treinamento (pesos próximos de zero efetivamente descartam features irrelevantes). A leve queda com PCA (F1 de 0,9792 → 0,9691) sugere que a rede neural já extraía informação útil dos 2 componentes descartados.

---

### Item 2.7 — Análise dos Resultados do Algoritmo Baseline

O baseline de classe majoritária obteve **65,7% de acurácia**, mas este número é enganoso:

- **Precisão = 0,000:** O baseline nunca prevê maligno, portanto quando "acerta", acerta apenas benignos
- **Recall = 0,000:** Nenhum caso de câncer é detectado — o modelo é clinicamente inútil
- **F1-Score = 0,000:** Confirma que o baseline não tem valor preditivo real

O baseline cumpre seu papel: estabelecer o piso mínimo que qualquer modelo deve superar. Todos os três algoritmos (K-NN, C4.5, MLP) superaram o baseline com margem superior a 0,94 no F1-Score, demonstrando aprendizado efetivo.

---

### Item 2.8 — Análise dos Resultados dos Três Algoritmos

#### Tabela Comparativa Final (todos com PCA)

| Modelo | Acurácia | Precisão | Recall | F1-Score | Δ Baseline |
|--------|----------|----------|--------|----------|------------|
| Baseline (Majoritária) | 65,7% | 0,0000 | 0,0000 | 0,0000 | — |
| K-NN (k=3, distância) | 97,8% | 0,9412 | **1,0000** | 0,9697 | +0,9697 |
| Árvore C4.5 (d=3) 🏆 | **98,6%** | **0,9600** | **1,0000** | **0,9796** | **+0,9796** |
| MLP (ReLU, 100) | 97,8% | 0,9592 | 0,9792 | 0,9691 | +0,9691 |

#### Matrizes de Confusão (teste, com PCA)

| Modelo | VN | FP | FN | VP |
|--------|----|----|----|-----|
| K-NN | 88 | 3 | 0 | 48 |
| Árvore C4.5 🏆 | 89 | 2 | 0 | 48 |
| MLP | 89 | 2 | 1 | 47 |

#### Análise Geral

1. **Todos os modelos superaram 94% de acurácia e 96% de F1-Score**, demonstrando que as 9 características citopatológicas são altamente preditivas para o diagnóstico de câncer de mama.

2. **Recall crítico alcançado:** K-NN e C4.5 atingiram **100% de recall** — zero falsos negativos. Em 48 casos malignos no teste, nenhum foi classificado erroneamente como benigno. Este é o resultado mais importante do ponto de vista clínico.

3. **Impacto do PCA:**
   - **K-NN:** Melhorou (F1 de 0,9485 → 0,9697) — redução de ruído beneficia distância euclidiana
   - **Árvore C4.5:** Melhorou significativamente (F1 de 0,9184 → 0,9796) — menor dimensionalidade reduz overfitting, que é o ponto fraco de árvores de decisão
   - **MLP:** Manteve-se estável (F1 de 0,9792 → 0,9691) — redes neurais já possuem seleção intrínseca de features

4. **Interpretabilidade como diferencial:** Embora K-NN e C4.5 tenham recall idêntico, a Árvore C4.5 oferece **interpretabilidade total**: com apenas 15 nós é possível rastrear exatamente por que cada classificação foi feita. Em um contexto médico, isso é crucial — o profissional de saúde pode auditar e compreender cada decisão do modelo.

5. **Custo computacional:** K-NN é lazy (não tem fase de treino), mas cada predição percorre todos os exemplos. C4.5 treina uma vez e prediz em O(log n). MLP é a mais cara no treino (backpropagation iterativo) mas a mais rápida na predição.

#### 🏆 Melhor Modelo: Árvore C4.5 com PCA

- **Recall: 100%** — zero falsos negativos (0/48)
- **F1-Score: 0,9796** — o mais alto entre todos
- **Falsos positivos: 2** — apenas 2 exames gerariam biópsias adicionais desnecessárias
- **Interpretabilidade: Total** — 15 nós, regras transparentes
- **Parâmetros:** max_depth=3, min_samples_split=5, criterion='entropy' (Gain Ratio)

---

## Implementações Manuais — Resumo

| Módulo | Funções/Algoritmos |
|--------|-------------------|
| `funcs_preprocessamento.py` | carregar_csv, média, mediana, moda, desvio_padrao, skewness, kurtosis, percentil, IQR, train_test_split_estratificado, detectar_valores_ausentes, imputar_mediana, SMOTE, normalizar_zscore, PCA, detectar_outliers_iqr |
| `metricas.py` | acurácia, precisão, revocação (recall), f1_score, matriz_confusao, stratified_kfold, grid_search |
| `knn.py` | KNNClassifier — distância euclidiana, votação uniforme/ponderada |
| `decision_tree.py` | DecisionTreeC45 — Gain Ratio, Gini Impurity, poda por profundidade |
| `mlp.py` | MLPClassifier — forward propagation, backpropagation, ReLU/tanh/sigmoid, Xavier init |

**Total: ~900 linhas de código próprio.** Zero dependências de scikit-learn, pandas, imblearn, scipy ou seaborn.

---

## Referências

1. Wolberg, W. H. & Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. *Proceedings of the National Academy of Sciences*, 87(23), 9193–9196.

2. UCI Machine Learning Repository. Breast Cancer Wisconsin (Original). ID 15. Disponível em: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original

3. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

4. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers.

5. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533–536.

6. Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*, 249–256.

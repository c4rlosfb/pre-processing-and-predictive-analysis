<h1 align="center">Breast Cancer Wisconsin 🩺 | ML from Scratch</h1>

<p align="center">
  <i>Análise preditiva e classificação de tumores utilizando algoritmos de Machine Learning implementados 100% do zero em Python.</i>
</p>

## 📌 Sobre o Projeto

Este projeto foi desenvolvido para a disciplina de Inteligência Artificial no IFSULDEMINAS (Campus Poços de Caldas). O objetivo principal é a classificação binária de tumores da base de dados Breast Cancer Wisconsin (Original) em benignos ou malignos.

O grande destaque técnico deste repositório é a construção manual de toda a pipeline de dados e predição: **todas as funções, métricas e algoritmos foram escritos do zero**, totalizando aproximadamente 900 linhas de código próprio. Nenhuma biblioteca de alto nível para machine learning (como `scikit-learn`, `pandas` ou `imblearn`) foi utilizada no processo.

## ⚙️ Arquitetura e Algoritmos

A arquitetura foi dividida em módulos limpos e especializados, utilizando apenas `numpy`, `math`, `csv` e `matplotlib` como dependências:

*   **`funcs_preprocessamento.py`:** Implementação de Z-score, imputação pela mediana, balanceamento de classes via SMOTE e redução de dimensionalidade utilizando PCA (Principal Component Analysis).
*   **`metricas.py`:** Cálculo manual de Matriz de Confusão, F1-Score, Acurácia, Recall, além de validação cruzada Stratified K-Fold combinada com Grid Search.
*   **`knn.py`:** Classificador K-Nearest Neighbors com cálculo de distância euclidiana e votação ponderada/uniforme.
*   **`decision_tree.py`:** Árvore de Decisão C4.5 com construção recursiva baseada em Gain Ratio (Entropia) e Gini Impurity, incluindo poda por profundidade.
*   **`mlp.py`:** Rede Neural Multilayer Perceptron com *forward* e *backpropagation* manuais, inicialização Xavier e funções de ativação ReLU, Tanh e Sigmoid.

## 📊 Destaques da Análise e Resultados

Durante a fase exploratória, optou-se estrategicamente por não remover outliers (método IQR), pois no contexto clínico oncológico, valores extremos em características citopatológicas representam os casos mais informativos. A aplicação do PCA reduziu a base de 9 para 7 componentes ortogonais, retendo 96,4% da variância.

**🏆 Melhor Modelo: Árvore C4.5 (com PCA)**
Em diagnósticos de câncer, o erro mais grave é o falso negativo (não detectar a doença). O modelo vencedor obteve resultados excepcionais no conjunto de teste:
*   **Recall:** 100% (Zero falsos negativos em 48 casos malignos).
*   **F1-Score:** 0,9796.
*   **Acurácia:** 98,6%.
*   **Interpretabilidade:** Alta transparência de decisão com apenas 15 nós.

## 🚀 Como Executar

Como o projeto foi construído puramente com bibliotecas base, a execução no terminal (seja no Linux, WSL ou Git Bash) é bastante simples:

```bash
# Clone o repositório
git clone [https://github.com/c4rlosfb/pre-processing-and-predictive-analysis.git](https://github.com/c4rlosfb/pre-processing-and-predictive-analysis.git)

# Acesse o diretório do projeto
cd pre-processing-and-predictive-analysis

# (Opcional) Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate

# Execute a rotina de treinamento e avaliação
python treinar_modelos.py
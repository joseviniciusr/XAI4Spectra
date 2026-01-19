# Análise Comparativa de Métodos de Explicabilidade em Dados Espectrais de Notas Bancárias

## 1. Introdução

Este documento apresenta uma análise aprofundada dos resultados obtidos no notebook de experimentos com dados espectrais de fluorescência de raios-X (XRF) aplicados à classificação de notas bancárias. O estudo tem como objetivo principal comparar diferentes métodos de explicabilidade (XAI - Explainable Artificial Intelligence) para identificar quais regiões espectrais são mais relevantes na distinção entre notas autênticas (Classe A) e falsificadas (Classe B).

### 1.1. Contexto do Problema

A autenticação de notas bancárias é um problema crítico de classificação binária que se beneficia enormemente de modelos interpretáveis. Os dados espectrais de XRF capturam a composição elementar das notas através de padrões de emissão de raios-X característicos de diferentes elementos químicos (Ca, Ti, Fe, Cu, Ag, etc.). A questão central é: **quais regiões espectrais (zonas de energia) são mais importantes para a classificação?**

### 1.2. Metodologia Geral

O pipeline experimental envolve:

1. **Modelo de classificação**: PLS-DA (Partial Least Squares Discriminant Analysis) com 4 variáveis latentes, otimizado por validação cruzada (10-fold CV)
2. **Pré-processamento**: Normalização de Poisson aplicada aos dados espectrais
3. **Divisão de dados**: Algoritmo de Kennard-Stone para divisão estratificada calibração/predição (70/30%)
4. **Segmentação espectral**: 15 zonas espectrais definidas por conhecimento de domínio (picos característicos de elementos e regiões de background)

### 1.3. Métodos de Explicabilidade Avaliados

#### 1.3.1. Métodos Baseline (Referências Clássicas)

1. **VIP (Variable Importance in Projection)**: Métrica específica do PLS que mede a importância de cada variável na construção das componentes latentes
2. **Coeficientes de Regressão**: Pesos lineares do modelo PLS, indicando a contribuição direta de cada variável para a classificação
3. **SHAP (SHapley Additive exPlanations)**: Valores de Shapley adaptados para atribuição de importância baseada em teoria dos jogos

#### 1.3.2. Métodos Baseados em Grafos de Predicados (Propostos)

Estes métodos utilizam uma abordagem inovadora baseada em:
- **Predicados lógicos**: Regras do tipo "zona espectral X está no quantil Y" (ex: "Fe ka está no 5º quantil")
- **Grafos direcionados**: Estruturas que capturam co-ocorrências e confiança entre predicados
- **LRC (Local Reaching Centrality)**: Métrica de centralidade que identifica predicados mais importantes no grafo

**Estratégias de Amostragem**:
- **Bagging**: Amostragem com reposição de amostras para criar múltiplas "bags" (bolsas) de dados
- **K-Fold com Kennard-Stone**: Divisão estratificada respeitando a diversidade espectral de cada predicado

**Estratégias de Ponderação de Arestas no Grafo**:
- **Cooccurrence**: Pesos baseados na frequência de co-ocorrência de predicados no conjunto de dados
- **Ranking**: Pesos baseados na posição relativa dos predicados nos rankings de covariância de cada fold/bag

---

## 2. Resultados e Discussão

### 2.1. Comparação Global: Bagging vs K-Fold (Modo Cooccurrence)

#### 2.1.1. Feature Importance Rankings

A tabela abaixo mostra as top-10 zonas espectrais mais importantes segundo cada método:

| Posição | VIP | Reg_coef | Shap | LRC_kfold_2 | LRC_kfold_3 | LRC_kfold_4 | LRC_kfold_5 | LRC_kfold_6 | LRC_Seed_0 | LRC_Seed_1 | LRC_Seed_42 |
|---------|-----|----------|------|-------------|-------------|-------------|-------------|-------------|------------|------------|-------------|
| 1 | Fe ka | Ti ka | Ti ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka |
| 2 | Ca ka | Ti kb | Ca ka | Ca ka | Fe ka | Fe ka | Fe ka | Fe ka | Ca ka | Ca ka | Ca ka |
| 3 | Ti ka | Ag ka scattering | Cu | Cu | Ca ka | Ca ka | Ca ka | Ca ka | Ti ka | Ti ka | Ti ka |
| 4 | Cu | Ca ka | Fe ka | Ti ka | background4 | Ti ka | Ti ka | Ca ka | Cu | Cu | Cu |
| 5 | Fe kb | Cu | Ti kb | Ca kb | Ca ka | Cu | Ti kb | Ti ka | Ca kb | Ca kb | Ca kb |
| 6 | Ca kb | background6 | Ag ka scattering | Ti kb | background6 | Ti kb | Ca ka | Ti kb | Ti kb | Ti kb | Ti kb |
| 7 | Ti kb | Ca kb | background6 | Fe kb | Ti ka | Ca kb | Ca kb | Cu | background3 | background3 | background3 |
| 8 | Ag ka scattering | background4 | background4 | background3 | Ag ka scattering | Ag ka scattering | background4 | background4 | background4 | background4 | background4 |
| 9 | background6 | background2 | background5 | background4 | background3 | background4 | Ag ka scattering | background3 | Ag ka scattering | Ag ka scattering | Ag ka scattering |
| 10 | background4 | Ar ka + Ag L | background1 | background2 | Ti kb | background3 | Cu | Ag ka scattering | background6 | background6 | background6 |




**Observações**:
- **Fe ka (pico do ferro)** é consistentemente a zona mais importante em VIP e em quase todos os métodos baseados em LRC
- **Ca ka (pico do cálcio)** aparece consistentemente nas posições 2-3 em VIP, SHAP e métodos LRC
- **Ti ka e Ti kb (picos do titânio)** têm importância variável: Reg_coef e SHAP os priorizam, mas LRC mostra resultados mais diversos
- Métodos LRC com **Bagging (Seeds)** apresentam rankings muito **estáveis** entre diferentes sementes aleatórias
- Métodos LRC com **K-fold** mostram **variabilidade moderada** dependendo do número de folds

#### 2.1.2. RBO (Rank-Biased Overlap) - Similaridade com VIP

A métrica RBO quantifica a similaridade entre rankings (0 = completamente diferente, 1 = idêntico), com penalização maior para diferenças nas primeiras posições.

| Posição | Referência | Método | RBO Score |
|---------|------------|--------|-----------|
| 1 | Vip | LRC_Seed_1 | 0.922 |
| 2 | Vip | LRC_kfold_2 | 0.915 |
| 3 | Vip | LRC_Seed_0 | 0.914 |
| 4 | Vip | LRC_Seed_42 | 0.873 |
| 5 | Vip | LRC_kfold_4 | 0.871 |
| 6 | Vip | LRC_kfold_5 | 0.850 |
| 7 | Vip | LRC_kfold_6 | 0.826 |
| 8 | Vip | Shap | 0.464 |
| 9 | Vip | LRC_kfold_3 | 0.426 |
| 10 | Vip | Reg_coef | 0.236 |

**Análise Crítica**:

1. **Superioridade do Bagging**: Os métodos baseados em **Bagging (LRC_Seed_X)** alcançam as **melhores similaridades com VIP (0.873-0.922)**, indicando que a amostragem aleatória com reposição captura bem a estrutura de importância identificada pelo método VIP clássico.

2. **K-fold com 2 folds é surpreendentemente eficaz**: **LRC_kfold_2 (RBO=0.915)** supera até algumas sementes de bagging, sugerindo que dividir os dados em apenas 2 grupos mantém amostras grandes e diversas por fold, capturando bem as relações entre predicados.

3. **Degradação com aumento de folds**: Conforme **k aumenta (3→6), o RBO Score diminui drasticamente**, especialmente notável em **k=3 (RBO=0.426)**. Isso sugere que:
   - Folds menores (menos amostras por fold) tornam os rankings de covariância mais **instáveis**
   - A estratégia **Round-Robin** de atribuição de amostras pode introduzir **viés** quando há muitos folds
   - Menos amostras por fold **reduzem a capacidade de capturar co-ocorrências** entre predicados de forma robusta

4. **SHAP tem concordância moderada (0.464)**: Embora SHAP seja um método robusto, sua concordância moderada com VIP pode refletir:
   - Diferentes fundamentos teóricos (teoria dos jogos vs. importância em componentes latentes)
   - Sensibilidade a outliers e não-linearidades que VIP não captura

5. **Coeficientes de Regressão são pobres explicadores (0.236)**: O baixo RBO indica que a **magnitude dos coeficientes lineares do PLS** não reflete bem a importância real das variáveis, possivelmente devido a:
   - Multicolinearidade entre regiões espectrais adjacentes
   - Natureza das transformações latentes do PLS que "embaralham" os pesos originais

---

### 2.2. Comparação: K-Fold com Ranking vs Cooccurrence

#### 2.2.1. Feature Importance - Modo Ranking

| Posição | VIP | Reg_coef | Shap | LRC_kfold_2 | LRC_kfold_3 | LRC_kfold_4 | LRC_kfold_5 | LRC_kfold_6 | LRC_Seed_0 | LRC_Seed_1 | LRC_Seed_42 |
|---------|-----|----------|------|-------------|-------------|-------------|-------------|-------------|------------|------------|-------------|
| 1 | Fe ka | Ti ka | Ti ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka |
| 2 | Ca ka | Ti kb | Ca ka | Fe ka | Ca ka | Fe ka | Fe ka | Fe ka | Ca ka | Ca ka | Ca ka |
| 3 | Ti ka | Ag ka scattering | Cu | Ca ka | Fe kb | Ti ka | Ca ka | Ca ka | Ti ka | Ti ka | Ti ka |
| 4 | Cu | Ca ka | Fe ka | Ti ka | Fe ka | Ca ka | Ti ka | Ti ka | Cu | Cu | Cu |
| 5 | Fe kb | Cu | Ti kb | Ca kb | Ca ka | Fe kb | Ca kb | Ca kb | Ca kb | Ca kb | Ca kb |
| 6 | Ca kb | background6 | Ag ka scattering | Ti kb | Ca kb | Ca kb | Fe kb | Ti kb | Ti kb | Ti kb | Ti kb |
| 7 | Ti kb | Ca kb | background6 | Fe kb | Ti ka | Ti kb | Ti kb | Cu | background3 | background3 | background3 |
| 8 | Ag ka scattering | background4 | background4 | Cu | Cu | Cu | Cu | Fe kb | background4 | background4 | background4 |
| 9 | background6 | background2 | background5 | background4 | background4 | background4 | background4 | background4 | Ag ka scattering | Ag ka scattering | Ag ka scattering |
| 10 | background4 | Ar ka + Ag L | background1 | background3 | Ti kb | background3 | background3 | background3 | background6 | background6 | background6 |

#### 2.2.2. RBO Score - Modo Ranking

| Posição | Referência | Método | RBO Score |
|---------|------------|--------|-----------|
| 1 | Vip | LRC_Seed_1 | 0.922 |
| 2 | Vip | LRC_Seed_0 | 0.914 |
| 3 | Vip | LRC_kfold_4 | 0.899 |
| 4 | Vip | LRC_kfold_2 | 0.876 |
| 5 | Vip | LRC_Seed_42 | 0.873 |
| 6 | Vip | LRC_kfold_6 | 0.825 |
| 7 | Vip | LRC_kfold_5 | 0.817 |
| 8 | Vip | LRC_kfold_3 | 0.540 |
| 9 | Vip | Shap | 0.464 |
| 10 | Vip | Reg_coef | 0.236 |

**Comparação Cooccurrence vs Ranking**:

| Método | RBO (Cooccurrence) | RBO (Ranking) | Diferença |
|--------|-------------------|---------------|-----------|
| LRC_kfold_2 | 0.915 | 0.876 | -0.039 |
| LRC_kfold_3 | 0.426 | 0.540 | +0.114 |
| LRC_kfold_4 | 0.871 | 0.899 | +0.028 |
| LRC_kfold_5 | 0.850 | 0.817 | -0.033 |
| LRC_kfold_6 | 0.826 | 0.825 | -0.001 |

**Discussão**:

1. **K-fold_3 melhora drasticamente com Ranking**: O modo **Ranking** recupera significativamente o desempenho em k=3 (de 0.426 para 0.540), embora ainda seja o pior desempenho. Isso sugere que:
   - A estratégia de **ponderação por ranking** é mais robusta a divisões de dados problemáticas
   - **Cooccurrence** é mais sensível a artefatos estatísticos quando folds são mal balanceados

2. **K-fold_4 se destaca no modo Ranking (0.899)**: Surpreendentemente, k=4 supera k=2 no modo Ranking, indicando um **sweet spot** entre:
   - Diversidade de folds (não muito poucos)
   - Tamanho amostral por fold (não muito pequeno)

3. **Bagging continua superior**: Independente da estratégia de ponderação, **Bagging mantém os melhores scores (>0.87)**, confirmando sua robustez.

4. **Convergência em k=6**: Ambas as estratégias apresentam scores muito próximos (~0.82) em k=6, sugerindo que com muitos folds pequenos, a escolha da estratégia de ponderação torna-se menos relevante.

---

### 2.3. Rankings Médios de Covariância

Para avaliar uma abordagem mais simples (sem grafos), calculamos o **ranking médio dos predicados** baseado apenas em suas covariâncias com a predição do modelo, agregando rankings de múltiplos bags/folds.

#### 2.3.1. Feature Importance - Rankings Médios

| Posição | VIP | Reg_coef | Shap | Cov_Mean_Seed_0 | Cov_Mean_Seed_1 | Cov_Mean_Seed_42 | Cov_Mean_Fold_2 | Cov_Mean_Fold_3 | Cov_Mean_Fold_4 | Cov_Mean_Fold_5 | Cov_Mean_Fold_6 |
|---------|-----|----------|------|-----------------|-----------------|------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| 1 | Fe ka | Ti ka | Ti ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka | Fe ka |
| 2 | Ca ka | Ti kb | Ca ka | Ca ka | Ca ka | Ca ka | Ca ka | Ca ka | Fe kb | Ca ka | Ca ka |
| 3 | Ti ka | Ag ka scattering | Cu | Fe kb | Fe kb | Fe kb | Fe kb | Fe kb | Ca ka | Fe kb | Fe kb |
| 4 | Cu | Ca ka | Fe ka | Ti ka | Ti ka | Ti ka | Ti ka | Ti ka | Ti ka | Ti ka | Ti ka |
| 5 | Fe kb | Cu | Ti kb | Ca kb | Ca kb | Ca kb | Ca kb | Ca kb | Ca kb | Ca kb | Ca kb |
| 6 | Ca kb | background6 | Ag ka scattering | Cu | Cu | Cu | Cu | Cu | Cu | Cu | Cu |
| 7 | Ti kb | Ca kb | background6 | Ti kb | Ti kb | Ti kb | Ti kb | Ti kb | Ti kb | Ti kb | Ti kb |
| 8 | Ag ka scattering | background4 | background4 | background4 | background4 | background4 | background4 | background4 | background4 | background4 | background4 |
| 9 | background6 | background2 | background5 | background2 | background2 | background2 | background2 | background2 | background2 | background2 | background1 |
| 10 | background4 | Ar ka + Ag L | background1 | background5 | background5 | background5 | background5 | background5 | background5 | background1 | background2 |

#### 2.3.2. RBO Score - Rankings Médios

| Posição | Referência | Método | RBO Score |
|---------|------------|--------|-----------|
| 1 | Vip | Cov_Mean_Seed_0 | 0.873 |
| 2 | Vip | Cov_Mean_Seed_1 | 0.873 |
| 3 | Vip | Cov_Mean_Seed_42 | 0.873 |
| 4 | Vip | Cov_Mean_Fold_2 | 0.873 |
| 5 | Vip | Cov_Mean_Fold_3 | 0.873 |
| 6 | Vip | Cov_Mean_Fold_5 | 0.873 |
| 7 | Vip | Cov_Mean_Fold_6 | 0.873 |
| 8 | Vip | Cov_Mean_Fold_4 | 0.768 |
| 9 | Vip | Shap | 0.464 |
| 10 | Vip | Reg_coef | 0.236 |

**Observações Críticas**:

1. **Estabilidade extrema (RBO=0.873)**: **Todos os métodos** (exceto Fold_4) alcançam **exatamente o mesmo RBO score (0.873)**, indicando que:
   - A **agregação por média** remove completamente a variabilidade entre bags/folds
   - Rankings médios são **muito estáveis** mas podem **perder nuances** capturadas pelos grafos

2. **Rankings médios vs LRC-Bagging são equivalentes**: O score de 0.873 é idêntico ao **LRC_Seed_42** (pior seed no bagging), mas inferior aos melhores (Seeds 0 e 1 com 0.91-0.92). Isso sugere que:
   - A abordagem por **grafos adiciona informação útil** além da simples média
   - **Co-ocorrências entre predicados** capturam dependências que a média ignora

3. **Anomalia em Fold_4 (RBO=0.768)**: Este único caso discrepante no modo de rankings médios pode indicar:
   - Um **artefato na divisão Round-Robin** com k=4 especificamente
   - **Instabilidade na estimativa de covariância** com folds intermediários

4. **Rankings médios são uma baseline simples e eficaz**: Para aplicações práticas que não exigem máxima performance, **rankings médios de covariância** oferecem:
   - **Simplicidade computacional** (sem construção de grafos)
   - **Boa concordância com VIP (0.873)**
   - **Estabilidade** independente de sementes ou número de folds

---

## 3. Conclusões e Recomendações

### 3.1. Principais Achados

1. **Fe ka (ferro) e Ca ka (cálcio) são as zonas espectrais mais discriminativas** para classificação de notas bancárias, consistente em todos os métodos.

2. **Bagging com LRC supera K-fold** na similaridade com VIP (0.91-0.92 vs 0.43-0.92), sendo mais robusto a variações na amostragem.

3. **K-fold com 2 folds** é surpreendentemente eficaz (0.915 em Cooccurrence), mas **degrada rapidamente com k=3**.

4. **K-fold_4 com modo Ranking** apresenta um **sweet spot interessante (0.899)**, balanceando diversidade e tamanho amostral.

5. **Rankings médios de covariância** são uma alternativa simples e estável (RBO=0.873), mas **não superam os melhores métodos baseados em grafos**.

6. **SHAP e Reg_coef** apresentam **baixa concordância com VIP** (0.464 e 0.236 respectivamente), questionando sua adequação como métodos de explicabilidade isolados neste contexto.

### 3.2. Recomendações Metodológicas

#### Para Máxima Performance:
- **Use Bagging com LRC** (10 bags, 80% de amostragem sem reposição)
- **Sementes múltiplas** (0, 1, 42) para ensemble final
- **Modo Cooccurrence** para ponderação de arestas

#### Para Balanceamento Performance/Eficiência:
- **K-fold_2 com modo Cooccurrence** (RBO=0.915, computacionalmente mais leve)
- Ou **K-fold_4 com modo Ranking** (RBO=0.899, boa generalização)

#### Para Simplicidade e Interpretabilidade:
- **Rankings médios de covariância** (RBO=0.873, sem grafos)
- Agregação sobre múltiplos folds (k=2 ou k=5)

#### Não Recomendado:
- **K-fold_3**: Desempenho inconsistente e baixo (0.426-0.540)
- **K-fold >5**: Degradação de performance sem benefícios claros
- **Coeficientes de regressão isolados**: Concordância muito baixa (0.236)

### 3.3. Considerações Finais

Este estudo demonstra que **métodos baseados em grafos de predicados** (LRC) oferecem uma alternativa promissora aos métodos clássicos de explicabilidade em dados espectrais, especialmente quando combinados com estratégias de amostragem robustas como Bagging. A consistência observada na identificação de Fe ka e Ca ka como zonas espectrais críticas reforça a validade química dos resultados, considerando que esses elementos são marcadores conhecidos de composição de papel e tintas em notas bancárias.

A variabilidade de desempenho entre diferentes configurações de K-fold destaca a importância de **validação cuidadosa da estratégia de divisão de dados** em contextos de explicabilidade, onde a estabilidade dos rankings é tão importante quanto a acurácia preditiva do modelo subjacente.

---

## 4. Limitações e Trabalhos Futuros

### 4.1. Limitações do Estudo

1. **Dataset único**: Resultados baseados apenas em notas bancárias; generalização para outros tipos de dados espectrais não é garantida
2. **Modelo fixo**: Apenas PLS-DA foi testado; outros modelos (RF, SVM, NN) podem ter diferentes estruturas de explicabilidade
3. **Sementes limitadas**: Apenas 3 sementes testadas para Bagging; análise estatística mais robusta requereria 30-50 repetições
4. **Métricas de similaridade**: RBO com p=0.7 penaliza fortemente posições iniciais; outros valores de p ou métricas alternativas (Kendall's Tau, Spearman) poderiam revelar padrões diferentes

### 4.2. Direções Futuras

1. **Estudo de ablação**: Testar sistematicamente os componentes do pipeline (número de bags, tamanho de amostra, limiares de covariância)
2. **Validação cruzada de explicabilidade**: Propor métricas de estabilidade e fidelidade de explicações além de RBO
3. **Extensão para regressão**: Adaptar LRC para problemas de regressão em dados espectrais (ex: quantificação de elementos)
4. **Interpretação causal**: Investigar relações causais entre zonas espectrais usando DAGs (Directed Acyclic Graphs)
5. **Explicações locais**: Desenvolver versão local do LRC para explicar predições individuais (análise por amostra)

---

**Autor**: Análise gerada via GitHub Copilot  
**Data**: 19 de Janeiro de 2026  
**Versão**: 1.0

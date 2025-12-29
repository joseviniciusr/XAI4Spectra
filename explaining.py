def extract_spectral_zones(Xcal, cuts):
    """
    Extract spectral zones from a DataFrame based on specified cuts.
    
    Parameters
    ----------
    - **Xcal** : pd.DataFrame
        DataFrame with spectral data, where columns are wavelengths/energies.
    - **cuts** : list of tuples/lists or dicts
        Each item defines a spectral zone to extract.
        - If tuple/list: (start, end) or (name, start, end)
        - If dict: {'name': str, 'start': float, 'end': float}
    
    Returns
    -------
    - **zones** : dict
        Dictionary where keys are zone names and values are DataFrames with the extracted spectral zones.
    """
    import numpy as np
    import pandas as pd

    # convert the column names to numeric when possible (NaN when not convertible)
    col_nums = pd.to_numeric(Xcal.columns.astype(str), errors='coerce')
    zones = {} # dictionary to store extracted zones

    for cut in cuts:
        # normalize cut format
        if isinstance(cut, dict): # if dict
            name = cut.get('name', f"{cut.get('start')}-{cut.get('end')}") # default name if not provided
            start = cut.get('start') # getting start value
            end = cut.get('end') # getting end value
        elif isinstance(cut, (list, tuple)): # if list/tuple
            if len(cut) == 2: 
                start, end = cut # getting start and end values
                name = f"{start}-{end}" # default name
            elif len(cut) == 3: # if name provided
                name, start, end = cut # getting name, start and end values
            else:
                raise ValueError("Cuts in tuple/list format must have 2 or 3 elements.")
        else:
            raise ValueError("Each cut must be a dict or a tuple/list.")

        # validate start and end
        try:
            s = float(start)
            e = float(end)
        except Exception: # Exception for conversion errors
            raise ValueError("star and end must be numeric values (int/float or convertible strings).")

        if s > e: # swap if necessary
            s, e = e, s

        # to select columns whose numeric value is in the interval [s, e]
        mask = (~np.isnan(col_nums)) & (col_nums >= s) & (col_nums <= e)
        selected_cols = list(Xcal.columns[mask])

        # piecing the zone DataFrame into the dictionary
        zones[name] = Xcal.loc[:, selected_cols]

    return zones

def aggregate_spectral_zones(spectral_zones_dict, aggregator='sum'):
    """
    Agrega os valores das zonas espectrais usando diferentes funções de agregação.
    
    Esta função processa cada zona espectral (DataFrame com múltiplas colunas de energia)
    e reduz cada linha (amostra) a um único valor numérico usando a função de agregação
    especificada.
    
    Parameters
    ----------
    - **spectral_zones_dict** : dict
        Dicionário retornado por extract_spectral_zones, onde:
        - chaves = nomes das zonas espectrais (ex: 'Ca ka', 'Fe ka')
        - valores = DataFrames com dados espectrais (linhas=amostras, colunas=energias)
    
    - **aggregator** : str, opcional (padrão='sum')
        Função de agregação a aplicar nas colunas de cada zona. Opções:
        - **'sum'**: Soma de todos os valores da zona (padrão)
        - **'mean'**: Média aritmética dos valores
        - **'median'**: Mediana dos valores
        - **'max'**: Valor máximo na zona
        - **'std'**: Desvio padrão dos valores
        - **'var'**: Variância dos valores
    
    Returns
    -------
    - **aggregated_df** : pd.DataFrame
        DataFrame com valores agregados, onde:
        - linhas = amostras (mesmo índice dos DataFrames originais)
        - colunas = zonas espectrais
        - valores = resultado da agregação (mesmo formato que .sum(axis=1))
    
    Raises
    ------
    - ValueError
        Se o agregador especificado não for reconhecido.
    """
    import pandas as pd
    import numpy as np
    
    # VALIDAÇÃO DE ENTRADA
    valid_aggregators = ['sum', 'mean', 'median', 'max', 'min', 'std', 'var']
    
    if aggregator not in valid_aggregators:
        raise ValueError(
            f"Agregador '{aggregator}' não reconhecido.\n"
            f"Opções válidas: {', '.join(valid_aggregators)}"
        )
    
    # MAPEAMENTO DOS AGREGADORES
    # Dicionário que mapeia strings para funções do pandas
    aggregation_functions = {
        'sum': lambda df: df.sum(axis=1),        # soma ao longo das colunas
        'mean': lambda df: df.mean(axis=1),      # média
        'median': lambda df: df.median(axis=1),  # mediana
        'max': lambda df: df.max(axis=1),        # valor máximo
        'min': lambda df: df.min(axis=1),        # valor mínimo
        'std': lambda df: df.std(axis=1),        # desvio padrão
        'var': lambda df: df.var(axis=1),        # variância
    }
    
    # AGREGAÇÃO DAS ZONAS ESPECTRAIS
    aggregated_dict = {}  # dicionário para armazenar resultados
    
    for zone_name, zone_df in spectral_zones_dict.items():
        # Aplica a função de agregação selecionada
        # O resultado é uma Series (mesma estrutura que .sum(axis=1))
        aggregated_series = aggregation_functions[aggregator](zone_df)
        
        # Armazena no dicionário
        aggregated_dict[zone_name] = aggregated_series
    
    # CONSTRUÇÃO DO DATAFRAME FINAL
    # Cada chave vira uma coluna, preservando os índices originais
    aggregated_df = pd.DataFrame(aggregated_dict)    
    return aggregated_df

def predicates_by_quantiles(zone_sums_df, quantiles):
    """
    Generate predicates based on specified quantiles for each column in a DataFrame
    and create a predicate indicator matrix.
    
    Parameters
    ----------
    - **zone_sums_df** : pd.DataFrame
        DataFrame with summed values for spectral zones.
    - **quantiles** : list of float
        List of quantiles (between 0 and 1) to generate predicates for.
    
    Returns
    -------
    - **predicates_df** : pd.DataFrame
        DataFrame containing the generated predicates with columns:
        'predicate', 'rule', 'zone', 'thresholds', 'operator'.
    - **predicate_indicator_df** : pd.DataFrame
        Binary indicator matrix (samples × predicates) where 1 indicates
        the sample satisfies the predicate, 0 otherwise.
    """
    import pandas as pd
    import numpy as np

    # calculating the quantiles for each column of zone_sums_df
    zone_quantiles = zone_sums_df.quantile(quantiles)
    
    zone_predicate_list = []
    predicate_num = 1
    for zone in zone_sums_df.columns:
        for q in quantiles:
            q_value = zone_quantiles.loc[q, zone]
            # <= Q
            zone_predicate_list.append({
                'predicate': f'P{predicate_num}',
                'rule': f"{zone} <= {q_value:.2f}",
                'zone': zone,
                'thresholds': f"{q_value:.2f}",
                'operator': "<="
            })
            predicate_num += 1
            # > Q
            zone_predicate_list.append({
                'predicate': f'P{predicate_num}',
                'rule': f"{zone} > {q_value:.2f}",
                'zone': zone,
                'thresholds': f"{q_value:.2f}",
                'operator': ">"
            })
            predicate_num += 1
    
    predicates_df = pd.DataFrame(zone_predicate_list)
    
    # Generating the predicate indicator DataFrame
    
    # function to evaluate a predicate for a given value
    def eval_predicate(value, thresholds, operator):
        if operator == "<=":
            return float(value <= float(thresholds))
        elif operator == ">":
            return float(value > float(thresholds))
        else:
            return np.nan
    
    # compute all columns first, then concatenate them at once
    columns_dict = {}
    
    # iterating over each predicate
    for _, row in predicates_df.iterrows():
        pred = row['predicate']
        zone = row['zone']
        thresholds = row['thresholds']
        operator = row['operator']
        columns_dict[pred] = zone_sums_df[zone].apply(
            lambda v: eval_predicate(v, thresholds, operator)
        ).astype(int)
    
    # create DataFrame from all columns at once
    predicate_indicator_df = pd.DataFrame(columns_dict, index=zone_sums_df.index)
    
    # setting column names to rules for better readability
    predicate_indicator_df.columns = predicates_df['rule'].tolist()
    
    # computing co-occurrence matrix
    co_occurrence_matrix = np.dot(predicate_indicator_df.T, predicate_indicator_df)
    co_occurrence_matrix_df = pd.DataFrame(co_occurrence_matrix, index=predicate_indicator_df.columns, columns=predicate_indicator_df.columns) 

    return predicates_df, predicate_indicator_df, co_occurrence_matrix_df

def create_predicate_info_dict(predicates_df, predicate_indicator_df, zone_aggregated_df, y_predicted_numeric):
    """
    Cria um dicionário com informações detalhadas sobre cada predicado.
    
    Para cada predicado, armazena:
    - Os valores agregados da zona espectral correspondente (das amostras que satisfazem o predicado)
    - Os valores preditos pelo modelo (das mesmas amostras)
    - Opcionalmente: índices das amostras, classe predita, etc.
    
    Parameters
    ----------
    - **predicates_df** : pd.DataFrame
        DataFrame com predicados gerados por `predicates_by_quantiles()` ou similar.
        Colunas obrigatórias: ['predicate', 'rule', 'zone', 'thresholds', 'operator']
        
    - **predicate_indicator_df** : pd.DataFrame
        Matriz binária de indicadores (samples × predicates) retornada por `predicates_by_quantiles()`.
        Colunas são as regras dos predicados (ex: "Ca ka <= 25.5")
        Valores: 1 = amostra satisfaz o predicado, 0 = não satisfaz
        
    - **zone_aggregated_df** : pd.DataFrame
        DataFrame com valores agregados das zonas espectrais (retornado por `aggregate_spectral_zones()`).
        Linhas = amostras, Colunas = zonas espectrais
        Valores = resultado da agregação (sum, mean, median, std, etc.)
        
    - **y_predicted_numeric** : pd.Series, pd.DataFrame ou np.ndarray
        Valores preditos pelo modelo (contínuos).
        - Para PLS-DA: valores entre 0 e 1 (ex: `plsda_results[5].iloc[:, -1]`)
        - Para PLS-R: valores contínuos da variável resposta
        - Deve ter o mesmo número de linhas que `zone_aggregated_df`
    
    Returns
    -------
    - **predicate_info_dict** : dict
        Dicionário estruturado como:
        {
            'Ca ka <= 25.5': DataFrame({
                'Zone_Aggregated': [valores agregados da zona Ca ka],
                'Predicted_Y': [valores preditos pelo modelo],
                'Sample_Index': [índices originais das amostras]
            }),
            'Fe ka > 10.2': DataFrame({...}),
            ...
        }
        
        - Chaves: Regras dos predicados (strings)
        - Valores: DataFrames com 3 colunas:
            - **Zone_Aggregated**: Valores agregados da zona espectral (pode ser soma, média, mediana, etc.)
            - **Predicted_Y**: Valores preditos pelo modelo para essas amostras
            - **Sample_Index**: Índices originais das amostras (para rastreabilidade)
    
    Raises
    ------
    - ValueError
        Se os DataFrames de entrada tiverem número incompatível de amostras
    - KeyError
        Se alguma coluna obrigatória estiver faltando
    """
    import pandas as pd
    import numpy as np
    
    # VALIDAÇÃO DE ENTRADAS
    
    # Verificar colunas obrigatórias em predicates_df
    required_cols = ['predicate', 'rule', 'zone', 'thresholds', 'operator']
    missing_cols = [col for col in required_cols if col not in predicates_df.columns]
    if missing_cols:
        raise KeyError(f"Colunas faltando em predicates_df: {missing_cols}")
    
    # Converter y_predicted_numeric para Series se necessário
    if isinstance(y_predicted_numeric, pd.DataFrame):
        y_predicted_numeric = y_predicted_numeric.iloc[:, -1]  # última coluna
    elif isinstance(y_predicted_numeric, np.ndarray):
        y_predicted_numeric = pd.Series(y_predicted_numeric)
    
    # Verificar compatibilidade de tamanhos
    n_samples_zones = len(zone_aggregated_df)
    n_samples_predicted = len(y_predicted_numeric)
    n_samples_indicators = len(predicate_indicator_df)
    
    if not (n_samples_zones == n_samples_predicted == n_samples_indicators):
        raise ValueError(
            f"Número incompatível de amostras:\n"
            f"  zone_aggregated_df: {n_samples_zones}\n"
            f"  y_predicted_numeric: {n_samples_predicted}\n"
            f"  predicate_indicator_df: {n_samples_indicators}\n"
            f"Todos devem ter o mesmo número de linhas."
        )
    
    # CONSTRUÇÃO DO DICIONÁRIO DE INFORMAÇÕES
    
    predicate_info_dict = {}  # dicionário para armazenar resultados
    n_predicates_processed = 0  # contador de predicados processados
    n_predicates_empty = 0  # contador de predicados sem amostras
    
    # Iterar sobre cada predicado
    for _, row in predicates_df.iterrows():
        
        pred_rule = row['rule']  # regra do predicado (ex: "Ca ka <= 25.5")
        zone_name = row['zone']  # nome da zona espectral (ex: "Ca ka")
        
        # 1. IDENTIFICAR AMOSTRAS QUE SATISFAZEM O PREDICADO
        # Usar a matriz de indicadores para filtrar amostras
        # predicate_indicator_df tem colunas com as regras dos predicados
        
        if pred_rule not in predicate_indicator_df.columns:
            # Predicado não existe na matriz de indicadores (não deveria acontecer)
            continue
        
        # Máscara booleana: True = amostra satisfaz o predicado
        mask_satisfied = predicate_indicator_df[pred_rule] == 1
        
        # Índices das amostras que satisfazem o predicado
        # Usar np.where() para compatibilidade com todos os tipos de índices
        satisfied_indices = np.where(mask_satisfied)[0].tolist()
        
        # 2. VERIFICAR SE HÁ AMOSTRAS SATISFEITAS
        if not satisfied_indices:  # lista vazia
            n_predicates_empty += 1
            continue  # pula este predicado (não adiciona ao dicionário)
        
        # 3. EXTRAIR VALORES AGREGADOS DA ZONA ESPECTRAL
        # Valores agregados (soma, média, mediana, std, etc.) da zona correspondente
        zone_aggregated_values = zone_aggregated_df.loc[satisfied_indices, zone_name]
        
        # 4. EXTRAIR VALORES PREDITOS PELO MODELO
        predicted_values = y_predicted_numeric.iloc[satisfied_indices]
        
        # 5. CRIAR DATAFRAME COM INFORMAÇÕES DO PREDICADO
        df_predicate_info = pd.DataFrame({
            'Zone_Aggregated': zone_aggregated_values.reset_index(drop=True),  # valores agregados
            'Predicted_Y': predicted_values.reset_index(drop=True),  # valores preditos
            'Sample_Index': satisfied_indices  # índices originais (para rastreabilidade)
        })
        
        # 6. ARMAZENAR NO DICIONÁRIO
        predicate_info_dict[pred_rule] = df_predicate_info
        n_predicates_processed += 1
    
    return predicate_info_dict

def bagging_predicates(zone_sums_df, y_predicted_numeric, predicates_df, 
                          n_bags=50, n_predicates_per_bag=20, n_samples_per_bag=80, 
                          min_samples_per_predicate=5, replace=True, random_seed=42,
                          sample_bagging=True, predicate_bagging=True):
    """
    Realiza bagging de predicados com controle granular sobre amostragem.
    
    Estratégia de Bagging (Configurável):
    =====================================
    1. **Amostragem de Linhas (Amostras):**
       - sample_bagging=True: Sorteia N amostras para cada bag
       - sample_bagging=False: Usa TODAS as amostras em todos os bags
    
    2. **Amostragem de Colunas (Predicados):**
       - predicate_bagging=True: Sorteia M predicados para cada bag
       - predicate_bagging=False: Usa TODOS os predicados em todos os bags
    
    3. **Filtragem e Validação:**
       - Para cada predicado selecionado, filtra as amostras que o satisfazem
       - Descarta predicados com insuficiente cobertura (se sample_bagging=True)
    
    Parâmetros
    ----------
    zone_sums_df : pd.DataFrame
        DataFrame com somas das zonas espectrais (linhas=amostras, colunas=zonas).
        
    y_predicted_numeric : pd.Series ou np.ndarray
        Valores preditos pelo modelo (contínuos, entre 0 e 1 para PLS-DA).
        
    predicates_df : pd.DataFrame
        DataFrame com predicados. Colunas obrigatórias:
        - 'rule': Regra legível (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
        
    n_bags : int, default=50
        Número de bags (iterações) a criar.
        
    n_predicates_per_bag : int, default=20
        Número de predicados a sortear por bag.
        **Ignorado se predicate_bagging=False.**
        
    n_samples_per_bag : int, default=80
        Número de amostras a sortear por bag.
        **Ignorado se sample_bagging=False.**
        
    min_samples_per_predicate : int, default=5
        Mínimo de amostras que devem satisfazer um predicado para ele ser válido.
        **Aplicado apenas se sample_bagging=True.**
        
    replace : bool, default=True
        - True: Bootstrap (amostragem com reposição)
        - False: Subsampling (sem reposição)
        **Aplicado apenas se sample_bagging=True.**
        
    random_seed : int, default=42
        Semente aleatória para reprodutibilidade.
        
    sample_bagging : bool, default=True
        - True: Faz subamostragem das LINHAS (amostras variam entre bags)
        - False: Usa todas as amostras em todos os bags
        
    predicate_bagging : bool, default=True
        - True: Faz subamostragem das COLUNAS (predicados variam entre bags)
        - False: Usa todos os predicados em todos os bags
    
    Returns
    -------
    bags_dict : dict
        Dicionário estruturado como:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
    
    """
    import numpy as np
    import pandas as pd
    
    # INICIALIZAÇÃO
    np.random.seed(random_seed)
    
    n_total_samples = len(zone_sums_df)
    predicate_rules = predicates_df['rule'].tolist()
    bags_dict = {}
    
    # LOOP PRINCIPAL: CRIAÇÃO DOS BAGS
    for bag_num in range(1, n_bags + 1):
        
        # 1. SELEÇÃO DE AMOSTRAS (LINHAS) - Controle via `sample_bagging`
        if sample_bagging:
            # Sorteia N amostras (bootstrap ou subsampling)
            bag_sample_indices = np.random.choice(
                range(n_total_samples),
                size=n_samples_per_bag,
                replace=replace  # True=bootstrap, False=subsampling
            )
        else:
            # Usa TODAS as amostras disponíveis
            bag_sample_indices = np.arange(n_total_samples)
        
        # 2. SELEÇÃO DE PREDICADOS (COLUNAS) - Controle via `predicate_bagging`
        if predicate_bagging:
            # Sorteia M predicados aleatoriamente (sem reposição)
            selected_predicate_rules = np.random.choice(
                predicate_rules,
                size=min(n_predicates_per_bag, len(predicate_rules)),
                replace=False
            )
        else:
            # Usa TODOS os predicados disponíveis
            selected_predicate_rules = predicate_rules
        
        # 3. FILTRAGEM E VALIDAÇÃO DE PREDICADOS
        bag_predicate_dict = {}
        n_discarded = 0
        
        for pred_rule in selected_predicate_rules:
            
            # Recupera metadados do predicado
            pred_row_filtered = predicates_df[predicates_df['rule'] == pred_rule]
            if len(pred_row_filtered) == 0:
                continue  # Predicado não encontrado, pula
            pred_row = pred_row_filtered.iloc[0]
            zone = pred_row['zone']
            threshold = float(pred_row['thresholds'])
            operator = pred_row['operator']
            
            # Extrai valores da zona para as amostras do bag
            zone_values_bag = zone_sums_df.loc[bag_sample_indices, zone].values
            
            # Aplica a regra do predicado
            if operator == "<=":
                mask_satisfied = zone_values_bag <= threshold
            elif operator == ">":
                mask_satisfied = zone_values_bag > threshold
            else:
                continue  # Operador inválido, pula
            
            # Filtra amostras que satisfazem o predicado
            satisfied_indices_in_bag = bag_sample_indices[mask_satisfied]
            
            # Validação de cobertura mínima (apenas se sample_bagging=True)
            if sample_bagging and len(satisfied_indices_in_bag) < min_samples_per_predicate:
                n_discarded += 1
                continue
            
            # Validação básica (descarta predicados vazios sempre)
            if len(satisfied_indices_in_bag) == 0:
                n_discarded += 1
                continue
            
            # Armazena dados do predicado válido
            df_predicate_info = pd.DataFrame({
                'Zone_Sum': zone_sums_df.loc[satisfied_indices_in_bag, zone].values,
                'Predicted_Y': y_predicted_numeric.iloc[satisfied_indices_in_bag].values,
                'Sample_Index': satisfied_indices_in_bag
            })
            
            bag_predicate_dict[pred_rule] = df_predicate_info
        
        # 4. ARMAZENAMENTO DO BAG
        if len(bag_predicate_dict) > 0:
            bags_dict[f'Bag_{bag_num}'] = bag_predicate_dict
            
            # Log informativo
            samp_str = "Sim" if sample_bagging else "Não"
            pred_str = f"Sim ({n_predicates_per_bag})" if predicate_bagging else "Não (Todos)"
            print(f"Bag_{bag_num} | Amostras: {samp_str} | Predicados: {pred_str} | "
                  f"Válidos: {len(bag_predicate_dict)} | Descartados: {n_discarded}")
        else:
            print(f"Bag_{bag_num}: VAZIO (todos os predicados descartados)")
    
    return bags_dict

def calculate_predicate_metrics(bags_result, metric='mutual_info', threshold=0.1, n_neighbors=10):
    """
    Calcula métricas de associação entre valores agregados das zonas espectrais 
    e as predições do modelo para cada predicado em cada bag.
    
    Esta função processa todos os bags gerados por `bagging_predicates()` e calcula
    a força da associação entre os valores das zonas espectrais e as predições contínuas
    do modelo. Suporta duas métricas: Mutual Information e Covariância.
    
    Parameters
    ----------
    - **bags_result** : dict
        Dicionário retornado por `bagging_predicates_v3()`, estruturado como:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
        
    - **metric** : str, opcional (padrão='mutual_info')
        Métrica de associação a calcular. Opções:
        - **'mutual_info'**: Informação Mútua (MI) - Mede dependência não-linear
        - **'covariance'**: Covariância - Mede dependência linear
        
    - **threshold** : float, opcional (padrão=0.1)
        Valor mínimo da métrica para um predicado ser considerado relevante.
        Predicados com métrica < threshold são FILTRADOS do resultado.
        - Para MI: valores típicos entre 0.0 e 1.0+ (quanto maior, mais informativo)
        - Para Covariance: valores dependem da escala dos dados (use valores absolutos)
        
    - **n_neighbors** : int, opcional (padrão=10)
        Número de vizinhos para o cálculo de Mutual Information.
        **Usado apenas quando metric='mutual_info'. Ignorado para covariância.**
        - Valores baixos (3-5): mais sensível a ruído local
        - Valores médios (10-20): balanço entre sensibilidade e robustez (recomendado)
        - Valores altos (>30): mais suave, menos sensível a variações locais
    
    Returns
    -------
    - **metrics_results_dict** : dict
        Dicionário estruturado como:
        {
            'Bag_1': DataFrame({
                'Predicate': ['Ca ka <= 25.5', 'Fe ka > 10.2', ...],
                'Mutual_Info': [0.45, 0.32, ...]  # ou 'Covariance' se metric='covariance'
            }),
            'Bag_2': DataFrame({...}),
            ...
        }
        
        Cada DataFrame contém:
        - **Predicate**: Regra do predicado (string)
        - **Mutual_Info** ou **Covariance**: Valor da métrica calculada
        - Ordenado de forma DECRESCENTE pela métrica (maiores valores primeiro)
        - Filtrado para manter apenas predicados com métrica > threshold
    
    Raises
    ------
    - ValueError
        Se metric não for 'mutual_info' ou 'covariance'
        
    - KeyError
        Se algum bag não contiver as colunas esperadas ('Zone_Sum', 'Predicted_Y')
    
    Notes
    -----
    - **Mutual Information (MI):**
        - Captura dependências LINEARES e NÃO-LINEARES entre X e Y
        - Valores sempre >= 0 (0 = independência, >0 = dependência)
        - Mais robusto a outliers que covariância
        - Computacionalmente mais custoso
        - Ideal para relações complexas/não-lineares
    
    - **Covariância:**
        - Captura apenas dependências LINEARES
        - Valores podem ser positivos ou negativos (usamos |valor absoluto|)
        - Sensível a outliers e escala dos dados
        - Computacionalmente mais rápido
        - Ideal para relações lineares simples
    
    - **Threshold:**
        - Define o "corte de relevância" para filtrar predicados fracos
        - Valores muito baixos: mantém muitos predicados (alguns irrelevantes)
        - Valores muito altos: pode descartar predicados úteis
        - Recomendação: começar com 0.1 para MI, ajustar conforme necessidade
    """
    import pandas as pd
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    
    # VALIDAÇÃO DE ENTRADAS    
    valid_metrics = ['mutual_info', 'covariance']
    if metric not in valid_metrics:
        raise ValueError(
            f"Métrica '{metric}' não reconhecida.\n"
            f"Opções válidas: {', '.join(valid_metrics)}"
        )
    
    if not isinstance(bags_result, dict):
        raise TypeError("bags_result deve ser um dicionário retornado por bagging_predicates_v3()")
    
    # INICIALIZAÇÃO    
    metrics_results_dict = {}  # dicionário para armazenar resultados
    metric_name = 'Mutual_Info' if metric == 'mutual_info' else 'Covariance'
    
    total_bags = len(bags_result)
    total_predicates_processed = 0
    total_predicates_filtered = 0
    
    print(f"\n{'='*70}")
    print(f"Calculando {metric_name} para Predicados")
    print(f"{'='*70}")
    print(f"Métrica: {metric}")
    print(f"Threshold: {threshold}")
    
    # LOOP PRINCIPAL: PROCESSAR CADA BAG    
    for bag_name, predicates_dict in bags_result.items():
        
        if len(predicates_dict) == 0:
            print(f"{bag_name}: VAZIO (pulando)")
            continue
        
        # 1. CALCULAR MÉTRICA PARA CADA PREDICADO NO BAG        
        metrics = {}  # dicionário temporário {predicate_rule: metric_value}
        
        for pred_rule, df_info in predicates_dict.items():
            
            # Validar colunas necessárias
            required_cols = ['Zone_Sum', 'Predicted_Y']
            missing_cols = [col for col in required_cols if col not in df_info.columns]
            if missing_cols:
                raise KeyError(
                    f"Bag '{bag_name}', Predicado '{pred_rule}': "
                    f"Colunas faltando: {missing_cols}"
                )
            
            # Extrair dados
            X_zone = df_info['Zone_Sum'].values.reshape(-1, 1)  # valores da zona (2D para sklearn)
            y_pred = df_info['Predicted_Y'].values  # valores preditos (1D)
            
            # Verificar se há dados suficientes
            if len(X_zone) < 2:
                metrics[pred_rule] = 0.0  # não há dados suficientes para calcular métrica
                continue
            
            # Calcular métrica selecionada
            if metric == 'mutual_info':
                # Mutual Information (não-linear)
                mi_score = mutual_info_regression(
                    X_zone, 
                    y_pred, 
                    discrete_features=False,  # X é contínua
                    n_neighbors=n_neighbors,
                    random_state=42  # reprodutibilidade
                )
                metrics[pred_rule] = mi_score[0]  # MI retorna array de 1 elemento
                
            elif metric == 'covariance':
                # Covariância (linear) - usamos valor absoluto
                # np.cov retorna matriz 2x2: [[var(X), cov(X,Y)], [cov(Y,X), var(Y)]]
                # Queremos cov(X,Y) = elemento [0,1] ou [1,0]
                cov_matrix = np.cov(X_zone.flatten(), y_pred)
                covariance = cov_matrix[0, 1]  # covariância X-Y
                metrics[pred_rule] = np.abs(covariance)  # valor absoluto
        
        total_predicates_processed += len(metrics)
        
        # 2. CONVERTER PARA DATAFRAME E ORDENAR        
        metrics_df = pd.DataFrame.from_dict(
            metrics, 
            orient='index',  # chaves = índices, valores = coluna
            columns=[metric_name]
        )
        
        # Adicionar coluna de predicado
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Ordenar de forma DECRESCENTE (maiores valores = mais informativos)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # 3. FILTRAR POR THRESHOLD        
        n_before_filter = len(metrics_df)
        metrics_df = metrics_df[metrics_df[metric_name] > threshold].reset_index(drop=True)
        n_after_filter = len(metrics_df)
        n_filtered = n_before_filter - n_after_filter
        
        total_predicates_filtered += n_filtered
        
        # 4. ARMAZENAR RESULTADO        
        metrics_results_dict[bag_name] = metrics_df
    
    return metrics_results_dict

def build_predicate_graph(bags_result, mi_results_dict, co_occurrence_matrix_df, 
                         predicates_df, random_state=42, show_details=True):
    """
    Constrói um grafo direcionado de predicados a partir dos resultados de bagging.
    
    Esta função cria um grafo onde:
    - Nós = predicados (regras) + nós terminais de classe
    - Arestas = transições entre predicados (ponderadas por co-ocorrência)
    - Direção = ordenada pela Mutual Information (MI)
    - Arestas bidirecionais são resolvidas mantendo a de maior peso
    
    Parameters
    ----------
    - **bags_result** : dict
        Dicionário com bags de predicados, estruturado como:
        {
            'Bag_1': {
                'Ca ka <= 25.5': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'Fe ka > 10.2': DataFrame([...]),
                ...
            },
            'Bag_2': {...},
            ...
        }
        
    - **mi_results_dict** : dict
        Dicionário com rankings de MI para cada bag:
        {'Bag_1': DataFrame(['Predicate', 'Mutual_Info']), 'Bag_2': ...}
        
    - **co_occurrence_matrix_df** : pd.DataFrame
        Matriz de co-ocorrência entre predicados (simétrica, predicados × predicados).
        Valores indicam quantas amostras satisfazem cada par de predicados.
        
    - **predicates_df** : pd.DataFrame
        DataFrame com informações dos predicados. Colunas obrigatórias:
        - 'rule': Regra do predicado (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
        
    - **random_state** : int, default=42
        Semente para desempate aleatório de arestas bidirecionais com mesmo peso.
    
    Returns
    -------
    - **DG** : nx.DiGraph
        Grafo direcionado contendo:
        - **Nós**: Predicados + nós terminais ('Class_1', 'Class_2')
        - **Atributos dos nós**: 
            - 'node_type': 'predicate' ou 'terminal'
            - 'class_label': 'eut' ou 'dist' (apenas para terminais)
        - **Arestas**: Direcionadas, com atributos:
            - 'weight': Peso acumulado (co-ocorrência)
            - 'bag': Nome do último bag que adicionou a aresta
    
    Notes
    -----
    **Estratégia de Construção:**
    1. Para cada bag, ordena predicados por MI (maior → menor)
    2. Cria caminho sequencial: P1 → P2 → P3 → ... → Terminal
    3. Acumula pesos de arestas repetidas entre bags diferentes
    4. Resolve arestas bidirecionais (A↔B) mantendo a de maior peso
    5. Conecta último predicado de cada caminho ao nó terminal da classe majoritária
    
    **Resolução de Bidirecionais:**
    - Se peso(A→B) > peso(B→A): Remove B→A
    - Se peso(B→A) > peso(A→B): Remove A→B
    - Se empate: Escolha aleatória (usa random_state)
    """
    import networkx as nx
    import numpy as np
    import pandas as pd
    
    np.random.seed(random_state)  # Para desempate aleatório
    
    # INICIALIZAÇÃO DO GRAFO    
    DG = nx.DiGraph()
    
    # Adicionar nós terminais
    DG.add_node('Class_A', node_type='terminal', class_label='A')
    DG.add_node('Class_B', node_type='terminal', class_label='B')
    
    # ACUMULAÇÃO DE ARESTAS    
    for bag_name, bag_predicates_dict in bags_result.items():
        
        # Obter ranking de MI para este bag
        mi_ranking = mi_results_dict[bag_name]
        ordered_predicates = mi_ranking['Predicate'].tolist()
        
        # Filtrar apenas predicados que existem neste bag
        ordered_predicates = [p for p in ordered_predicates if p in bag_predicates_dict.keys()]
        
        if len(ordered_predicates) == 0:
            continue
        
        # Construir caminho no grafo
        for i in range(len(ordered_predicates) - 1):
            pred_current = ordered_predicates[i]
            pred_next = ordered_predicates[i + 1]
            
            # Adicionar nós
            DG.add_node(pred_current, node_type='predicate')
            DG.add_node(pred_next, node_type='predicate')
            
            # Peso da aresta = co-ocorrência
            co_occurrence_raw = co_occurrence_matrix_df.loc[pred_current, pred_next]
            # Garantir que seja escalar
            if isinstance(co_occurrence_raw, (pd.Series, pd.DataFrame)):
                co_occurrence = float(co_occurrence_raw.iloc[0] if isinstance(co_occurrence_raw, pd.Series) else co_occurrence_raw.iloc[0, 0])
            else:
                co_occurrence = float(co_occurrence_raw)
            
            # Acumulação de peso
            if DG.has_edge(pred_current, pred_next):
                DG[pred_current][pred_next]['weight'] += co_occurrence
            else:
                DG.add_edge(pred_current, pred_next, weight=co_occurrence, bag=bag_name)
        
        
        # Conectar último predicado ao terminal
        last_pred = ordered_predicates[-1]
        DG.add_node(last_pred, node_type='predicate') # o add_node aqui é para garantir que o nó do último predicado exista
        # esse passo pode ser redundante, mas garante que o nó esteja presente no grafo antes de criar a aresta para o terminal
        # Por ex casos de bags com um único predicado nao gerariam o nó do último predicado no loop anterior
        
        df_last = bag_predicates_dict[last_pred]
        class_counts = df_last['Class_Predicted'].value_counts()
        majority_class = class_counts.idxmax()
        terminal_node = f'Class_{majority_class}'
        n_samples_last = class_counts.sum()
        
        if DG.has_edge(last_pred, terminal_node):
            DG[last_pred][terminal_node]['weight'] += n_samples_last
        else:
            DG.add_edge(last_pred, terminal_node, weight=n_samples_last, bag=bag_name)

    # IDENTIFICAÇÃO E REMOÇÃO DE ARESTAS BIDIRECIONAIS    
    bidirectional_pairs = []
    processed = set()
    
    for u, v in DG.edges():
        if DG.has_edge(v, u) and (v, u) not in processed:
            weight_forward_raw = DG[u][v]['weight']
            weight_reverse_raw = DG[v][u]['weight']
            
            # Garantir que sejam escalares
            if isinstance(weight_forward_raw, (pd.Series, pd.DataFrame)):
                weight_forward = float(weight_forward_raw.iloc[0] if isinstance(weight_forward_raw, pd.Series) else weight_forward_raw.iloc[0, 0])
            else:
                weight_forward = float(weight_forward_raw)
                
            if isinstance(weight_reverse_raw, (pd.Series, pd.DataFrame)):
                weight_reverse = float(weight_reverse_raw.iloc[0] if isinstance(weight_reverse_raw, pd.Series) else weight_reverse_raw.iloc[0, 0])
            else:
                weight_reverse = float(weight_reverse_raw)
            
            bidirectional_pairs.append({
                'node_A': u,
                'node_B': v,
                'weight_A_to_B': weight_forward,
                'weight_B_to_A': weight_reverse
            })
            
            processed.add((u, v))
            processed.add((v, u))
    print(f"\nTotal de pares bidirecionais encontrados: {len(bidirectional_pairs)}")       
    # Mostrar top 10 pares bidirecionais
    # if len(bidirectional_pairs) > 0:
    #     print("Top 10 pares bidirecionais (com pesos):")
    #     for pair in bidirectional_pairs[:10]:
    #         print(f"  {pair['node_A']} <--> {pair['node_B']} | "
    #               f"Peso {pair['node_A']}→{pair['node_B']}: {pair['weight_A_to_B']} | "
    #               f"Peso {pair['node_B']}→{pair['node_A']}: {pair['weight_B_to_A']}")

    n_removed = 0
    # Remover arestas perdedoras
    for pair in bidirectional_pairs:
        u = pair['node_A']
        v = pair['node_B']
        weight_forward = pair['weight_A_to_B']
        weight_reverse = pair['weight_B_to_A']
        
        if weight_forward > weight_reverse:
            DG.remove_edge(v, u)
            print(f"Removida aresta {v} -> {u} (peso {weight_reverse})") if show_details else None
            print(f"Mantida aresta {u} -> {v} (peso {weight_forward})\n") if show_details else None
            print("="*70 + "\n") if show_details else None
            n_removed += 1
        elif weight_reverse > weight_forward:
            DG.remove_edge(u, v)
            print(f"Removida aresta {u} -> {v} (peso {weight_forward})") if show_details else None
            print(f"Mantida aresta {v} -> {u} (peso {weight_reverse})\n") if show_details else None
            print("="*70 + "\n") if show_details else None
            n_removed += 1
        else:
            # Empate: escolha aleatória
            if np.random.rand() > 0.5:
                DG.remove_edge(v, u)
                print(f"Empate! Removida aresta {v} -> {u} (peso {weight_reverse})") if show_details else None
                print(f"Mantida aresta {u} -> {v} (peso {weight_forward})\n") if show_details else None
                print("="*70 + "\n") if show_details else None
                n_removed += 1
            else:
                DG.remove_edge(u, v)
                print(f"Empate! Removida aresta {u} -> {v} (peso {weight_forward})") if show_details else None
                print(f"Mantida aresta {v} -> {u} (peso {weight_reverse})\n") if show_details else None
                print("="*70 + "\n") if show_details else None
                n_removed += 1

    # resumo final do grafo
    print(f"\nTotal de arestas iniciais: {DG.number_of_edges() + n_removed}")
    print(f"Total de arestas removidas por bidirecionalidade: {n_removed}")
    print(f"Arestas bidirecionais restantes: {len(bidirectional_pairs) - n_removed}")
    print(f"Total de nós predicados: {len([n for n, attr in DG.nodes(data=True) if attr['node_type'] == 'predicate'])}")
    print(f"Total de nós terminais: {len([n for n, attr in DG.nodes(data=True) if attr['node_type'] == 'terminal'])}\n")      
    
    return DG

def calculate_lrc(graphs_by_seed, predicates_df):
    """
    Calcula Local Reaching Centrality (LRC) para todos os nós dos grafos.
    
    A LRC mede a importância de cada nó baseada em sua capacidade de alcançar
    outros nós no grafo, ponderada pelos pesos das arestas. Nós com maior LRC
    são mais centrais/importantes na estrutura do grafo.
    
    Parameters
    ----------
    - **graphs_by_seed** : dict
        Dicionário com grafos NetworkX por semente (retornado por build_predicate_graphs).
        Estrutura: {seed1: nx.DiGraph(), seed2: nx.DiGraph(), ...}
        
    - **predicates_df** : pd.DataFrame
        DataFrame com informações dos predicados. Colunas obrigatórias:
        - 'rule': Regra do predicado (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
    
    Returns
    -------
    - **lrc_by_seed** : dict
        Dicionário com DataFrames de LRC para cada semente:
        {
            seed1: DataFrame(['Node', 'Local_Reaching_Centrality', 'Zone', 'Threshold', 'Operator', 'Seed']),
            seed2: DataFrame([...]),
            ...
        }
        
        Cada DataFrame contém:
        - **Node**: Nome do nó (regra do predicado ou 'Class_eut'/'Class_dist')
        - **Local_Reaching_Centrality**: Valor da LRC (quanto maior, mais importante)
        - **Zone**: Nome da zona espectral (None para nós terminais)
        - **Threshold**: Valor do threshold (None para nós terminais)
        - **Operator**: Operador da regra (None para nós terminais)
        - **Seed**: Semente aleatória usada
        
        **Ordenação**: Decrescente por LRC (nós mais importantes primeiro)
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
        
    # CÁLCULO DA LRC
    
    lrc_by_seed = {}
    
    for seed, DG in graphs_by_seed.items():
        print(f"\nProcessando LRC - Semente: {seed}")
        
        # 1. CALCULAR LRC PARA CADA NÓ
        local_reaching_centrality = {
            node: nx.local_reaching_centrality(DG, node, weight='weight')
            for node in DG.nodes()
        }
        
        # Ordenar por LRC (decrescente)
        sorted_lrc = sorted(
            local_reaching_centrality.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 2. CRIAR DATAFRAME COM LRC
        lrc_df_seed = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
        
        # 3. EXTRAIR METADADOS DOS PREDICADOS
        zones = []
        thresholds = []
        operators = []
        
        for node in lrc_df_seed['Node']:
            if node.startswith('Class_'):
                # Nó terminal
                zones.append(None)
                thresholds.append(None)
                operators.append(None)
            else:
                # Predicado: buscar metadados em predicates_df
                pred_row_filtered = predicates_df[predicates_df['rule'] == node]
                
                if len(pred_row_filtered) == 0:
                    # Predicado não encontrado (não deveria acontecer)
                    zones.append('Unknown')
                    thresholds.append(None)
                    operators.append(None)
                else:
                    pred_row = pred_row_filtered.iloc[0]
                    zones.append(pred_row['zone'])
                    thresholds.append(pred_row['thresholds'])
                    operators.append(pred_row['operator'])
        
        # Adicionar colunas ao DataFrame
        lrc_df_seed['Zone'] = zones
        lrc_df_seed['Threshold'] = thresholds
        lrc_df_seed['Operator'] = operators
        lrc_df_seed['Seed'] = seed
        
        # Armazenar resultado
        lrc_by_seed[seed] = lrc_df_seed
    
        return lrc_by_seed
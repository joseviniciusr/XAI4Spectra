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
        - **'min'**: Valor mínimo na zona
        - **'std'**: Desvio padrão dos valores
        - **'var'**: Variância dos valores
        - **'extreme'**: Valor de maior magnitude (mais intenso) na zona, ou seja,
          escolhe o valor com maior valor absoluto em cada amostra (pode ser positivo ou negativo)
    
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
    valid_aggregators = ['sum', 'mean', 'median', 'max', 'min', 'std', 'var', 'extreme']
    
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
        # 'extreme': escolhe o valor com maior magnitude (abs), preservando o sinal
        'extreme': lambda df: df.apply(
            lambda row: (row.loc[row.abs().idxmax()] if row.notna().any() else np.nan),
            axis=1
        ),
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
    
    # Removing duplicate predicates based on 'rule' column
    # Some zones may have the same quantile values, creating duplicate rules
    initial_count = len(predicates_df)
    predicates_df = predicates_df.drop_duplicates(subset=['rule'], keep='first').reset_index(drop=True)
    final_count = len(predicates_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate predicates. Remaining: {final_count}")
    
    # Renumbering predicates after removing duplicates
    predicates_df['predicate'] = [f'P{i+1}' for i in range(len(predicates_df))]

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
    
    print(f"Calculando {metric_name} para Predicados")
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

def calculate_lrc_single_graph(graph, predicates_df):
    """
    Calcula Local Reaching Centrality (LRC) para todos os nós de um único grafo.
    
    A LRC mede a importância de cada nó baseada em sua capacidade de alcançar
    outros nós no grafo, ponderada pelos pesos das arestas. Nós com maior LRC
    são mais centrais/importantes na estrutura do grafo.
    
    Parameters
    ----------
    - **graph** : nx.DiGraph
        Grafo direcionado do NetworkX (retornado por build_fold_predicate_graph ou similar).
        
    - **predicates_df** : pd.DataFrame
        DataFrame com informações dos predicados. Colunas obrigatórias:
        - 'rule': Regra do predicado (ex: "Ca ka <= 25.5")
        - 'zone': Nome da zona espectral
        - 'thresholds': Valor do threshold
        - 'operator': "<=" ou ">"
    
    Returns
    -------
    - **lrc_df** : pd.DataFrame
        DataFrame com as seguintes colunas:
        - **Node**: Nome do nó (regra do predicado ou 'Class_A'/'Class_B')
        - **Local_Reaching_Centrality**: Valor da LRC (quanto maior, mais importante)
        - **Zone**: Nome da zona espectral (None para nós terminais)
        - **Threshold**: Valor do threshold (None para nós terminais)
        - **Operator**: Operador da regra (None para nós terminais)
        
        **Ordenação**: Decrescente por LRC (nós mais importantes primeiro)
    """
    import networkx as nx
    import pandas as pd
    import numpy as np
    
    print(f"\nProcessando LRC do grafo...")
    
    # 1. CALCULAR LRC PARA CADA NÓ
    local_reaching_centrality = {
        node: nx.local_reaching_centrality(graph, node, weight='weight')
        for node in graph.nodes()
    }
    
    # Ordenar por LRC (decrescente)
    sorted_lrc = sorted(
        local_reaching_centrality.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 2. CRIAR DATAFRAME COM LRC
    lrc_df = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
    
    # 3. EXTRAIR METADADOS DOS PREDICADOS
    zones = []
    thresholds = []
    operators = []
    
    for node in lrc_df['Node']:
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
    lrc_df['Zone'] = zones
    lrc_df['Threshold'] = thresholds
    lrc_df['Operator'] = operators
    
    return lrc_df
    
def build_fold_predicate_graph(bags_result, mi_results_dict, predicates_df, 
                         random_state=42, show_details=True, 
                         normalize_weights=False,
                         weight_mode='ranking',
                         co_occurrence_matrix=None,
                         apply_confidence_multiplier=False,
                         accumulate_cooccurrence_weights=False):
    """
    Constrói um grafo direcionado de predicados com diferentes esquemas de peso.
    
    MODOS DE PESO DISPONÍVEIS (weight_mode):
    
    1. weight_mode='ranking' (PADRÃO):
       - Peso da aresta = posição invertida no ranking do predicado de ORIGEM
       - Predicados mais importantes (rank 1) contribuem com maior peso
       - Se mesma aresta aparece em múltiplos folds: SOMA os pesos
       - Desempate bidirecional: peso acumulado maior vence
       - Opção normalize_weights disponível
    
    2. weight_mode='cooccurrence':
       - Peso da aresta = número de amostras que satisfazem AMBOS os predicados
       - Valor SEMPRE obtido da matriz de co-ocorrência GLOBAL
       
       DUAS SUB-ESTRATÉGIAS DISPONÍVEIS:
       
       a) accumulate_cooccurrence_weights=False (PADRÃO):
          - Peso da aresta = valor da matriz (fixo)
          - Se aresta aparece em múltiplos folds: NÃO soma (mantém peso original)
          - Desempate bidirecional: usa score de confiança
       
       b) accumulate_cooccurrence_weights=True:
          - Peso da aresta = Σ valor_matriz para cada fold que contém a aresta
          - Se aresta aparece em múltiplos folds: SOMA o valor da matriz
          - Desempate bidirecional: usa peso acumulado
       
       Em ambos casos: apply_confidence_multiplier pode ser usado
    
    MULTIPLICADOR DE CONFIANÇA (apply_confidence_multiplier):
    
    Quando weight_mode='cooccurrence' e apply_confidence_multiplier=True:
    - Após resolver arestas bidirecionais, multiplica cada peso pelo seu score
    - Peso_final = co_ocorrência × score_de_confiança
    - Combina informação de amostras (co-ocorrência) com consistência (score)
    - Funciona com accumulate=False (peso fixo × score) ou accumulate=True (peso acumulado × score)
    
    SCORE DE CONFIANÇA:
    
    Usado para desempate de arestas bidirecionais quando peso é simétrico:
    - Cada vez que a aresta A→B aparece em um fold, soma-se o rank invertido de A
    - Score(A→B) = Σ (n_predicados - rank_A + 1) para cada fold onde A→B aparece
    - A direção com maior score de confiança vence
    - NOTA: Quando accumulate_cooccurrence_weights=True, usa peso acumulado ao invés de score
    
    OPÇÕES DE NORMALIZAÇÃO (apenas para weight_mode='ranking'):
    
    - normalize_weights=False (padrão):
      * Peso = (k - rank + 1), onde k = número de predicados
      * Rank 1 → peso = k (máximo) | Rank k → peso = 1 (mínimo)
    
    - normalize_weights=True:
      * Peso = (k - rank + 1) / k  (normalização no intervalo [1/k, 1])
      * Rank 1 → peso = 1.0 (máximo) | Rank k → peso = 1/k (mínimo positivo)
    
    Parameters
    ----------
    - bags_result : dict - Dicionário com folds de predicados
    - mi_results_dict : dict - Dicionário com rankings de MI/Cov para cada fold
    - predicates_df : pd.DataFrame - DataFrame com informações dos predicados
    - random_state : int - Semente para desempate aleatório (usado apenas em empates)
    - show_details : bool - Se True, imprime detalhes da resolução de bidirecionais
    - normalize_weights : bool - Se True, normaliza pesos no intervalo [0, 1]
                                 (APENAS para weight_mode='ranking')
    - weight_mode : str - 'ranking' ou 'cooccurrence'
    - co_occurrence_matrix : pd.DataFrame - Matriz de co-ocorrência global entre predicados
                                            (OBRIGATÓRIO se weight_mode='cooccurrence')
    - apply_confidence_multiplier : bool - Se True, multiplica peso final pelo score
                                           (APENAS para weight_mode='cooccurrence')
    - accumulate_cooccurrence_weights : bool - Se True, soma valores da matriz quando aresta
                                               aparece em múltiplos folds
                                               (APENAS para weight_mode='cooccurrence')
    
    Returns
    -------
    - DG : nx.DiGraph - Grafo direcionado com pesos e scores de confiança
    """
    import networkx as nx
    import numpy as np
    import pandas as pd
    
    # VALIDAÇÃO DE PARÂMETROS
    
    # Valida o modo de peso
    valid_modes = ['ranking', 'cooccurrence']
    if weight_mode not in valid_modes:
        raise ValueError(f"weight_mode deve ser um de {valid_modes}, recebido: '{weight_mode}'")
    
    # Se modo é cooccurrence, matriz é SEMPRE obrigatória
    if weight_mode == 'cooccurrence' and co_occurrence_matrix is None:
        raise ValueError("co_occurrence_matrix é obrigatório quando weight_mode='cooccurrence'")
    
    # Aviso se normalize_weights=True mas modo é cooccurrence
    if weight_mode == 'cooccurrence' and normalize_weights:
        print(" AVISO: normalize_weights=True ignorado em weight_mode='cooccurrence'")
        print("   (Pesos de co-ocorrência representam contagens reais de amostras)")
    
    # Aviso se apply_confidence_multiplier=True mas modo não é cooccurrence
    if weight_mode != 'cooccurrence' and apply_confidence_multiplier:
        print(" AVISO: apply_confidence_multiplier=True ignorado em weight_mode='ranking'")
        print("   (Multiplicador de confiança só se aplica ao modo 'cooccurrence')")
    
    # Aviso se accumulate_cooccurrence_weights=True mas modo não é cooccurrence
    if weight_mode != 'cooccurrence' and accumulate_cooccurrence_weights:
        print(" AVISO: accumulate_cooccurrence_weights=True ignorado em weight_mode='ranking'")
        print("   (Acumulação de co-ocorrências só se aplica ao modo 'cooccurrence')")
    
    # Define semente para reprodutibilidade em caso de empates
    np.random.seed(random_state)
    
    # FASE 1: INICIALIZAÇÃO DO GRAFO    
    print(f"\n{'='*60}")
    print(f"CONSTRUÇÃO DO GRAFO DE PREDICADOS")
    print(f"Modo de peso: {weight_mode.upper()}")
    if weight_mode == 'cooccurrence':
        print(f"Fonte: Matriz de co-ocorrência global")
        if accumulate_cooccurrence_weights:
            print(f"Sub-estratégia: ACUMULATIVA (soma valores da matriz)")
        else:
            print(f"Sub-estratégia: NÃO-ACUMULATIVA (peso fixo da matriz)")
    print(f"{'='*60}")
    
    # Cria grafo direcionado vazio
    DG = nx.DiGraph()
    
    # Adiciona nós terminais (classes de saída)
    DG.add_node('Class_A', node_type='terminal', class_label='A')
    DG.add_node('Class_B', node_type='terminal', class_label='B')
    
    # FASE 2: ACUMULAÇÃO DE ARESTAS
    
    # Contador de folds processados
    n_folds_processed = 0
    
    # Itera sobre cada fold
    for bag_name, bag_predicates_dict in bags_result.items():
        
        # Obtém o ranking de MI/Cov para este fold
        mi_ranking = mi_results_dict[bag_name]
        
        # Lista de predicados ordenados por importância (maior → menor)
        ordered_predicates = mi_ranking['Predicate'].tolist()
        
        # Filtra apenas predicados que existem neste fold
        ordered_predicates = [p for p in ordered_predicates if p in bag_predicates_dict.keys()]
        
        # Pula se não houver predicados válidos
        if len(ordered_predicates) == 0:
            continue # o continue pula para o próximo fold caso este esteja vazio
        
        # Incrementa contador de folds
        n_folds_processed += 1
        
        # Número total de predicados neste fold (usado para calcular score/peso)
        n_predicates = len(ordered_predicates)
        
        # LOOP: Constrói caminho sequencial P1 → P2 → P3 → ... → Terminal
        
        for i in range(len(ordered_predicates) - 1):
            # Predicado atual (origem da aresta)
            pred_current = ordered_predicates[i]
            # Próximo predicado (destino da aresta)
            pred_next = ordered_predicates[i + 1]
            
            # Adiciona nós ao grafo (se já existirem, apenas atualiza atributos)
            DG.add_node(pred_current, node_type='predicate')
            DG.add_node(pred_next, node_type='predicate')
            
            # Posição no ranking (1-indexed): 1 = mais importante
            rank_position = i + 1
            
            # CÁLCULO DO SCORE DE CONFIANÇA
            # Usado no modo cooccurrence para desempate e no modo ranking como peso
            # Score = rank invertido = (n_predicados - posição + 1)
            # Rank 1 → score = n | Rank 2 → score = n-1 | ... | Rank n → score = 1
            confidence_score = n_predicates - rank_position + 1 # o + 1 é para ajustar o rank 1 ao score máximo n_predicados
            
            # MODO RANKING: Peso = Score de Confiança (com opção de normalização)
            if weight_mode == 'ranking':
                if normalize_weights:
                    # PESO NORMALIZADO: (k - rank + 1) / k → intervalo [1/k, 1]
                    # Garante que o peso mínimo seja 1/k (positivo) para LRC funcionar corretamente
                    edge_weight = (n_predicates - rank_position + 1) / n_predicates
                else:
                    # PESO NÃO NORMALIZADO: score de confiança diretamente
                    edge_weight = confidence_score
                
                # Acumula peso se aresta já existe, senão cria nova aresta
                if DG.has_edge(pred_current, pred_next):
                    # Aresta já existe: SOMA o peso (comportamento original)
                    DG[pred_current][pred_next]['weight'] += edge_weight
                    DG[pred_current][pred_next]['confidence_score'] += confidence_score
                    DG[pred_current][pred_next]['fold_contributions'].append({
                        'fold': bag_name, 
                        'rank': rank_position, 
                        'weight': edge_weight,
                        'score': confidence_score
                    })
                else:
                    # Nova aresta: cria com peso inicial
                    DG.add_edge(
                        pred_current, 
                        pred_next, 
                        weight=edge_weight,
                        confidence_score=confidence_score,
                        fold_contributions=[{
                            'fold': bag_name, 
                            'rank': rank_position, 
                            'weight': edge_weight,
                            'score': confidence_score
                        }]
                    )
            
            # MODO CO-OCORRÊNCIA: Peso = amostras que satisfazem ambos predicados
            elif weight_mode == 'cooccurrence':
                # Obtém peso da matriz de co-ocorrência GLOBAL
                # A matriz é simétrica, então cooc[A,B] = cooc[B,A]
                try:
                    cooc_weight = co_occurrence_matrix.loc[pred_current, pred_next]
                except KeyError:
                    # Se par não existe na matriz, peso = 0
                    cooc_weight = 0
                    print(f" ⚠️ Par não encontrado na matriz: ({pred_current}, {pred_next})")
                
                # SUB-ESTRATÉGIA: ACUMULATIVA
                if accumulate_cooccurrence_weights:
                    # Cada vez que a aresta aparece, SOMA o valor da matriz
                    if DG.has_edge(pred_current, pred_next):
                        # Aresta já existe: SOMA o peso da matriz (acumulação)
                        DG[pred_current][pred_next]['weight'] += cooc_weight
                        DG[pred_current][pred_next]['confidence_score'] += confidence_score
                        DG[pred_current][pred_next]['fold_contributions'].append({
                            'fold': bag_name, 
                            'rank': rank_position, 
                            'weight': cooc_weight,
                            'score': confidence_score
                        })
                    else:
                        # Nova aresta: cria com peso da matriz
                        DG.add_edge(
                            pred_current, 
                            pred_next, 
                            weight=cooc_weight,
                            confidence_score=confidence_score,
                            fold_contributions=[{
                                'fold': bag_name, 
                                'rank': rank_position, 
                                'weight': cooc_weight,
                                'score': confidence_score
                            }]
                        )
                
                # SUB-ESTRATÉGIA: NÃO-ACUMULATIVA
                else:
                    # Peso fixo da matriz (não acumula)
                    if DG.has_edge(pred_current, pred_next):
                        # Aresta já existe: NÃO soma peso (mantém original)
                        # Mas ACUMULA o score de confiança para desempate posterior
                        DG[pred_current][pred_next]['confidence_score'] += confidence_score
                        DG[pred_current][pred_next]['fold_contributions'].append({
                            'fold': bag_name, 
                            'rank': rank_position, 
                            'weight': cooc_weight,
                            'score': confidence_score
                        })
                    else:
                        # Nova aresta: cria com peso da co-ocorrência
                        DG.add_edge(
                            pred_current, 
                            pred_next, 
                            weight=cooc_weight,
                            confidence_score=confidence_score,
                            fold_contributions=[{
                                'fold': bag_name, 
                                'rank': rank_position, 
                                'weight': cooc_weight,
                                'score': confidence_score
                            }]
                        )
        
        # CONEXÃO DO ÚLTIMO PREDICADO AO TERMINAL
        
        # Obtém o último predicado (menor importância no fold)
        last_pred = ordered_predicates[-1]
        
        # Garante que o nó existe
        DG.add_node(last_pred, node_type='predicate')
        
        # Obtém DataFrame do último predicado para determinar classe majoritária
        df_last = bag_predicates_dict[last_pred]
        
        # Conta amostras por classe predita
        class_counts = df_last['Class_Predicted'].value_counts()
        
        # Determina classe majoritária
        majority_class = class_counts.idxmax()
        
        # Define nó terminal correspondente
        terminal_node = f'Class_{majority_class}'
        
        # PESO DA ARESTA PARA O TERMINAL
        if weight_mode == 'ranking':
            if normalize_weights:
                # Peso mínimo positivo (1/k) para LRC funcionar corretamente
                terminal_weight = 1.0 / n_predicates  # Último rank normalizado
            else:
                terminal_weight = 1  # Score mínimo (último rank)
            terminal_score = 1  # Score mínimo (último rank)
            
            # Acumula peso na aresta para o terminal
            if DG.has_edge(last_pred, terminal_node):
                DG[last_pred][terminal_node]['weight'] += terminal_weight
                DG[last_pred][terminal_node]['confidence_score'] += terminal_score
            else:
                DG.add_edge(last_pred, terminal_node, 
                           weight=terminal_weight, 
                           confidence_score=terminal_score,
                           fold_contributions=[])
        
        elif weight_mode == 'cooccurrence':
            # Para terminal, usamos contagem de amostras do predicado
            terminal_weight = len(df_last)
            terminal_score = 1  # Score mínimo (último rank)
            
            if DG.has_edge(last_pred, terminal_node):
                # NÃO soma peso, mas acumula score
                DG[last_pred][terminal_node]['confidence_score'] += terminal_score
            else:
                DG.add_edge(last_pred, terminal_node,
                           weight=terminal_weight,
                           confidence_score=terminal_score,
                           fold_contributions=[])
    
    print(f"\nFolds processados: {n_folds_processed}")
    print(f"Arestas criadas (antes de resolver bidirecionais): {DG.number_of_edges()}")
    
    # FASE 3: IDENTIFICAÇÃO E REMOÇÃO DE ARESTAS BIDIRECIONAIS
    
    # Lista para armazenar pares bidirecionais encontrados
    bidirectional_pairs = []
    
    # Set para rastrear pares já processados
    processed = set()
    
    # Itera sobre todas as arestas do grafo
    for u, v in DG.edges():
        # Verifica se existe aresta reversa (bidirecional)
        if DG.has_edge(v, u) and (v, u) not in processed:
            # Obtém atributos de ambas as direções
            weight_forward = float(DG[u][v]['weight'])
            weight_reverse = float(DG[v][u]['weight'])
            score_forward = float(DG[u][v]['confidence_score'])
            score_reverse = float(DG[v][u]['confidence_score'])
            
            # Armazena informações do par bidirecional
            bidirectional_pairs.append({
                'node_A': u,
                'node_B': v,
                'weight_A_to_B': weight_forward,
                'weight_B_to_A': weight_reverse,
                'score_A_to_B': score_forward,
                'score_B_to_A': score_reverse
            })
            
            # Marca ambas direções como processadas
            processed.add((u, v))
            processed.add((v, u))
    
    # Imprime total de pares bidirecionais encontrados
    print(f"\n{'='*60}")
    print(f"RESOLUÇÃO DE ARESTAS BIDIRECIONAIS")
    print(f"{'='*60}")
    print(f"Total de pares bidirecionais encontrados: {len(bidirectional_pairs)}")
    
    if weight_mode == 'ranking':
        print(f"Critério de desempate: PESO ACUMULADO (soma dos ranks invertidos)")
    else:
        if accumulate_cooccurrence_weights:
            print(f"Critério de desempate: PESO ACUMULADO (soma das co-ocorrências locais)")
        else:
            print(f"Critério de desempate: SCORE DE CONFIANÇA (soma dos ranks invertidos)")
            print(f"(Peso de co-ocorrência global é simétrico, então usamos score para decidir direção)")
    
    # Contador de arestas removidas
    n_removed = 0
    
    # Resolve cada par bidirecional
    for pair in bidirectional_pairs:
        u = pair['node_A']
        v = pair['node_B']
        weight_forward = pair['weight_A_to_B']
        weight_reverse = pair['weight_B_to_A']
        score_forward = pair['score_A_to_B']
        score_reverse = pair['score_B_to_A']
        
        # CRITÉRIO DE DECISÃO DEPENDE DO MODO E SUB-ESTRATÉGIA
        if weight_mode == 'ranking':
            # Modo ranking: usa peso acumulado como critério
            criterion_forward = weight_forward
            criterion_reverse = weight_reverse
            criterion_name = "peso"
        elif weight_mode == 'cooccurrence' and accumulate_cooccurrence_weights:
            # Modo cooccurrence ACUMULATIVO: usa peso acumulado como critério
            # (pesos locais de folds diferentes são somados, então não são simétricos)
            criterion_forward = weight_forward
            criterion_reverse = weight_reverse
            criterion_name = "peso"
        else:
            # Modo cooccurrence GLOBAL: usa score de confiança como critério
            # (peso de co-ocorrência global é simétrico)
            criterion_forward = score_forward
            criterion_reverse = score_reverse
            criterion_name = "score"
        
        # RESOLUÇÃO DO PAR BIDIRECIONAL
        if criterion_forward > criterion_reverse:
            # Critério A→B é maior: remove B→A
            DG.remove_edge(v, u)
            if show_details:
                print(f"\n[{u} ↔ {v}]")
                print(f"  ✗ Removida: {v} → {u} ({criterion_name}={criterion_reverse:.2f}, peso={weight_reverse:.2f})")
                print(f"  ✓ Mantida:  {u} → {v} ({criterion_name}={criterion_forward:.2f}, peso={weight_forward:.2f})")
            n_removed += 1
            
        elif criterion_reverse > criterion_forward:
            # Critério B→A é maior: remove A→B
            DG.remove_edge(u, v)
            if show_details:
                print(f"\n[{u} ↔ {v}]")
                print(f"  ✗ Removida: {u} → {v} ({criterion_name}={criterion_forward:.2f}, peso={weight_forward:.2f})")
                print(f"  ✓ Mantida:  {v} → {u} ({criterion_name}={criterion_reverse:.2f}, peso={weight_reverse:.2f})")
            n_removed += 1
            
        else:
            # EMPATE: escolha aleatória baseada em random_state
            if np.random.rand() > 0.5:
                DG.remove_edge(v, u)
                if show_details:
                    print(f"\n[{u} ↔ {v}]  EMPATE ({criterion_name}={criterion_forward:.2f})")
                    print(f"  ✗ Removida (aleatório): {v} → {u}")
                    print(f"  ✓ Mantida:  {u} → {v}")
            else:
                DG.remove_edge(u, v)
                if show_details:
                    print(f"\n[{u} ↔ {v}]  EMPATE ({criterion_name}={criterion_forward:.2f})")
                    print(f"  ✗ Removida (aleatório): {u} → {v}")
                    print(f"  ✓ Mantida:  {v} → {u}")
            n_removed += 1
    
    # FASE 4: APLICAÇÃO DO MULTIPLICADOR DE CONFIANÇA (OPCIONAL)
    # Esta fase só é executada quando:
    # - weight_mode == 'cooccurrence' E
    # - apply_confidence_multiplier == True
    #
    # O objetivo é combinar a informação de co-ocorrência (quantas amostras
    # satisfazem ambos predicados) com a informação de confiança (quão 
    # consistentemente essa direção aparece nos folds com alta importância).
    #
    # Peso_final = co_ocorrência × score_de_confiança
    
    if weight_mode == 'cooccurrence' and apply_confidence_multiplier:
        print(f"\n{'='*60}")
        print(f"APLICAÇÃO DO MULTIPLICADOR DE CONFIANÇA")
        print(f"{'='*60}")
        print(f"Fórmula: peso_final = co_ocorrência × score_de_confiança")
        
        # Contador de arestas modificadas
        n_multiplied = 0
        
        # Itera sobre todas as arestas restantes no grafo
        for u, v, data in DG.edges(data=True):
            # Obtém peso original (co-ocorrência) e score de confiança
            original_weight = data['weight']
            confidence_score = data['confidence_score']
            
            # Calcula novo peso multiplicado
            # Peso_final = co_ocorrência × score
            new_weight = original_weight * confidence_score
            
            # Atualiza o peso da aresta
            DG[u][v]['weight'] = new_weight
            
            # Armazena valores originais para referência
            DG[u][v]['original_cooccurrence'] = original_weight
            
            n_multiplied += 1
            
            # Mostra detalhes se solicitado
            if show_details and n_multiplied <= 10:
                print(f"  {u} → {v}: {original_weight} × {confidence_score} = {new_weight:.2f}")
        
        if show_details and n_multiplied > 10:
            print(f"  ... e mais {n_multiplied - 10} arestas")
        
        print(f"\nTotal de arestas com peso multiplicado: {n_multiplied}")
    
    # FASE 5: RESUMO FINAL DO GRAFO    
    print(f"\n{'='*60}")
    print(f"RESUMO FINAL DO GRAFO")
    print(f"{'='*60}")
    
    # Imprime informações sobre o esquema de pesos usado
    if weight_mode == 'ranking':
        if normalize_weights:
            print(f"\nEsquema de Pesos: RANKING NORMALIZADO")
            print(f"  Fórmula: peso = (k - rank + 1) / k")
            print(f"  Intervalo: [1/k, 1.0] (mínimo positivo)")
        else:
            print(f"\nEsquema de Pesos: RANKING NÃO NORMALIZADO")
            print(f"  Fórmula: peso = (k - rank + 1)")
            print(f"  Intervalo: [1, k] (valores inteiros)")
        print(f"  Arestas repetidas: SOMA de pesos")
        print(f"  Desempate bidirecional: Peso acumulado maior vence")
    else:
        print(f"\nEsquema de Pesos: CO-OCORRÊNCIA (matriz global)")
        print(f"  Fonte: Matriz de co-ocorrência calculada no dataset completo")
        print(f"  Significado: Número de amostras que satisfazem ambos predicados")
        
        if accumulate_cooccurrence_weights:
            print(f"\n  Tipo: ACUMULATIVA")
            print(f"  Fórmula: peso = Σ matriz_cooc[A,B] para cada fold que contém A→B")
            print(f"  Arestas repetidas: SOMA o valor da matriz (acumulação entre folds)")
            print(f"  Desempate bidirecional: Peso acumulado maior vence")
        else:
            print(f"\n  Tipo: NÃO-ACUMULATIVA")
            print(f"  Fórmula: peso = matriz_cooc[A,B] (valor fixo)")
            print(f"  Arestas repetidas: NÃO soma (mantém peso original da matriz)")
            print(f"  Desempate bidirecional: Score de confiança (soma ranks invertidos)")
        
        if apply_confidence_multiplier:
            print(f"\n  ✓ MULTIPLICADOR DE CONFIANÇA APLICADO")
            print(f"    Fórmula final: peso = co_ocorrência × score_de_confiança")
            print(f"    Combina informação de amostras (matriz) com consistência (score)")
        else:
            print(f"\n  ✗ Multiplicador de confiança NÃO aplicado")
            print(f"    (use apply_confidence_multiplier=True para habilitar)")
    
    print(f"\nEstatísticas do Grafo:")
    print(f"  Total de arestas iniciais: {DG.number_of_edges() + n_removed}")
    print(f"  Arestas removidas (bidirecionais): {n_removed}")
    print(f"  Arestas finais: {DG.number_of_edges()}")
    print(f"  Nós predicados: {len([n for n, attr in DG.nodes(data=True) if attr.get('node_type') == 'predicate'])}")
    print(f"  Nós terminais: {len([n for n, attr in DG.nodes(data=True) if attr.get('node_type') == 'terminal'])}")
    
    return DG

def calculate_predicate_ranking_mean(mi_results_dict, return_unique_zones=False):
    """
    Calcula ranking agregado de predicados baseado na posição média em múltiplos rankings.
    
    Parâmetros:
    -----------
    mi_results_dict : dict
        Dicionário onde cada chave é um fold/bag e cada valor é um DataFrame contendo
        uma coluna 'Predicate' com os predicados ordenados por importância.
    return_unique_zones : bool, default=False
        Se True, retorna também um DataFrame com apenas a primeira ocorrência de cada zona espectral.
        
    Retorna:
    --------
    ranking_predicate_mean : pd.DataFrame
        DataFrame com colunas:
        - Predicate: Nome completo do predicado
        - Mean_Score: Score médio baseado nas posições nos rankings
        - Count: Número de vezes que o predicado aparece nos rankings
        - Zone: Zona espectral extraída do predicado
        - Rule: Operador da regra ('>' ou '<=')
        - Threshold: Valor do limiar da regra
        
    ranking_predicate_mean_unique : pd.DataFrame (opcional)
        Retornado apenas se return_unique_zones=True. Contém apenas a primeira
        ocorrência de cada zona espectral (maior Mean_Score).
        
    Exemplo:
    --------
    >>> ranking_df = calculate_predicate_ranking_mean(mi_results_dict)
    >>> ranking_df, ranking_unique_df = calculate_predicate_ranking_mean(mi_results_dict, return_unique_zones=True)
    """
    import pandas as pd
    import numpy as np
    
    # Criar DataFrame padronizado para todos os rankings
    max_len = max(len(mi_df['Predicate']) for mi_df in mi_results_dict.values())
    padded_dict = {
        f'Predicate_{fold}': list(mi_df['Predicate']) + [None]*(max_len - len(mi_df['Predicate']))
        for fold, mi_df in mi_results_dict.items()
    }
    all_results = pd.DataFrame(padded_dict)
    
    # Calculando scores baseados na posição no ranking
    predicate_scores = {}
    
    for col in all_results.columns:
        # Pegar valores não nulos da coluna
        non_null_predicates = all_results[col].dropna()
        k = len(non_null_predicates)
        
        # Atribuir scores: primeiro = k, segundo = k-1, ..., último = 1
        for idx, predicate in enumerate(non_null_predicates):
            score = k - idx
            if predicate not in predicate_scores:
                predicate_scores[predicate] = []
            predicate_scores[predicate].append(score)
    
    # Calcular score médio para cada predicado
    ranking_data = []
    for predicate, scores in predicate_scores.items():
        mean_score = np.mean(scores)
        ranking_data.append({
            'Predicate': predicate,
            'Mean_Score': mean_score,
            'Count': len(scores)  # Número de vezes que o predicado aparece nos rankings
        })
    
    # Criar dataframe e ordenar por score médio (decrescente)
    ranking_predicate_mean = pd.DataFrame(ranking_data)
    ranking_predicate_mean = ranking_predicate_mean.sort_values(by='Mean_Score', ascending=False).reset_index(drop=True)
    
    # Função auxiliar para parsing de predicados
    def parse_predicate(predicate_str):
        """
        Extrai zona, operador e threshold de um predicado.
        Formato esperado: 'Zone_name > threshold' ou 'Zone_name <= threshold'
        """
        if '>' in predicate_str:
            parts = predicate_str.split('>')
            zone = parts[0].strip()
            threshold = float(parts[1].strip())
            rule = '>'
        elif '<=' in predicate_str:
            parts = predicate_str.split('<=')
            zone = parts[0].strip()
            threshold = float(parts[1].strip())
            rule = '<='
        else:
            # Caso não encontre operador conhecido
            zone = predicate_str
            threshold = None
            rule = None
        return zone, rule, threshold
    
    # Aplicar parsing e criar novas colunas
    ranking_predicate_mean[['Zone', 'Rule', 'Threshold']] = ranking_predicate_mean['Predicate'].apply(
        lambda x: pd.Series(parse_predicate(x))
    )
    
    if return_unique_zones:
        # Selecionar apenas zonas espectrais únicas (primeira ocorrência = maior score)
        ranking_predicate_mean_unique = ranking_predicate_mean.drop_duplicates(
            subset=['Zone'], keep='first'
        ).reset_index(drop=True)
        return ranking_predicate_mean, ranking_predicate_mean_unique
    
    return ranking_predicate_mean

# PERMUTATION IMPORTANCE PARA PREDICADOS - Implementação Completa
#
# Esta implementação calcula a importância de cada predicado usando Permutation
# Importance aplicada à zona espectral completa do predicado. A estratégia é
# permutar o bloco inteiro de colunas espectrais (não coluna por coluna) para
# avaliar o impacto real da zona no modelo.

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    get_scorer, accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Tuple, Optional, Union, Callable, Literal
import warnings


# Constantes para tipos de tarefa
TASK_CLASSIFICATION = 'classification'
TASK_REGRESSION = 'regression'

# Métricas de classificação suportadas
CLASSIFICATION_METRICS = {
    'accuracy', 'f1', 'f1_weighted', 'f1_macro', 'roc_auc'
}

# Métricas de regressão suportadas (compatíveis com sklearn scoring)
REGRESSION_METRICS = {
    'neg_root_mean_squared_error', 'neg_mean_squared_error', 
    'neg_mean_absolute_error', 'r2'
}


def get_zone_columns_from_predicate(
    predicate_rule: str,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    Xcal_columns: pd.Index
) -> List[str]:
    """
    Obtém as colunas espectrais correspondentes à zona de um predicado.
    
    Esta função identifica qual zona espectral está associada a um predicado
    e retorna a lista de colunas (energias) que compõem essa zona.
    
    Parameters
    ----------
    predicate_rule : str
        Regra do predicado (ex: 'F1 <= 10.5')
    predicates_df : pd.DataFrame
        DataFrame com informações dos predicados (colunas: 'rule', 'zone', etc.)
    spectral_cuts : list of tuples
        Lista de cortes espectrais no formato [(nome, inicio, fim), ...]
    Xcal_columns : pd.Index
        Índice das colunas do DataFrame de calibração (energias)
    
    Returns
    -------
    list
        Lista de nomes de colunas (strings) que compõem a zona espectral
    
    Raises
    ------
    ValueError
        Se a zona não for encontrada nos spectral_cuts
    KeyError
        Se o predicado não existir em predicates_df
    
    Example
    -------
    >>> zone_cols = get_zone_columns_from_predicate('F1 <= 10.5', predicates_df, spectral_cuts, Xcal.columns)
    >>> print(f"Zona contém {len(zone_cols)} colunas: {zone_cols[:3]}...")
    """
    # 1. Encontrar a zona associada ao predicado
    mask = predicates_df['rule'] == predicate_rule
    if not mask.any():
        raise KeyError(f"Predicado '{predicate_rule}' não encontrado em predicates_df")
    
    zone_name = predicates_df.loc[mask, 'zone'].values[0]
    
    # 2. Encontrar os limites da zona nos spectral_cuts
    zone_start, zone_end = None, None
    for cut in spectral_cuts:
        if len(cut) == 3:
            name, start, end = cut
        elif len(cut) == 2:
            start, end = cut
            name = f"{start}-{end}"
        else:
            continue
        
        if name == zone_name:
            zone_start, zone_end = float(start), float(end)
            break
    
    if zone_start is None or zone_end is None:
        raise ValueError(f"Zona '{zone_name}' não encontrada em spectral_cuts")
    
    # 3. Selecionar colunas dentro do intervalo
    # Converter nomes de colunas para numérico quando possível
    col_numeric = pd.to_numeric(Xcal_columns.astype(str), errors='coerce')
    
    # Máscara para colunas dentro do intervalo [zone_start, zone_end]
    mask_cols = (~np.isnan(col_numeric)) & (col_numeric >= zone_start) & (col_numeric <= zone_end)
    
    zone_columns = list(Xcal_columns[mask_cols])
    
    return zone_columns


def _infer_task_type(y: Union[pd.Series, np.ndarray], scoring: str) -> str:
    """
    Infere automaticamente o tipo de tarefa (classificação ou regressão).
    
    Parameters
    ----------
    y : pd.Series or np.ndarray
        Valores alvo
    scoring : str
        Métrica de scoring utilizada
    
    Returns
    -------
    str
        'classification' ou 'regression'
    """
    # Primeiro, verificar pela métrica de scoring
    if scoring in CLASSIFICATION_METRICS:
        return TASK_CLASSIFICATION
    elif scoring in REGRESSION_METRICS:
        return TASK_REGRESSION
    
    # Se a métrica não é reconhecida, inferir pelo tipo de y
    y_array = np.array(y).flatten()
    
    # Se y é string ou object, é classificação
    if y_array.dtype.kind in ['U', 'S', 'O']:  # Unicode, byte string, or object
        return TASK_CLASSIFICATION
    
    # Se y é numérico, verificar se parece com classes discretas
    if np.issubdtype(y_array.dtype, np.integer):
        unique_values = np.unique(y_array)
        # Se tem poucos valores únicos (<=10), provavelmente é classificação
        if len(unique_values) <= 10:
            return TASK_CLASSIFICATION
    
    # Se y é float, verificar se são valores discretos
    if np.issubdtype(y_array.dtype, np.floating):
        unique_values = np.unique(y_array)
        # Se tem poucos valores únicos e parecem inteiros, pode ser classificação
        if len(unique_values) <= 10 and np.allclose(y_array, y_array.astype(int)):
            return TASK_CLASSIFICATION
        return TASK_REGRESSION
    
    # Default para classificação (comportamento original)
    return TASK_CLASSIFICATION


def _convert_pls_prediction_to_class(y_pred_continuous, threshold=0.5, class_labels=('A', 'B')):
    """
    Converte predições contínuas do PLS para rótulos de classe.
    
    PLSRegression retorna valores contínuos. Para classificação binária,
    convertemos usando um threshold: valores >= threshold → classe positiva.
    
    Parameters
    ----------
    y_pred_continuous : np.ndarray
        Predições contínuas do modelo PLS
    threshold : float
        Limiar para classificação (padrão=0.5)
    class_labels : tuple
        Rótulos das classes (positiva, negativa)
    
    Returns
    -------
    np.ndarray
        Array de rótulos de classe
    """
    # Achatar array se necessário (PLS pode retornar shape (n, 1))
    y_flat = np.array(y_pred_continuous).flatten()
    
    # Converter para classes usando threshold
    return np.where(y_flat >= threshold, class_labels[0], class_labels[1])

def spectral_perturbation_importance(model, X, y_pred_original, spectral_cuts, 
                                      perturbation_value=0, metric='mean_abs_diff'):
    """
    Perturba regiões espectrais e avalia o impacto nas predições do modelo.
    
    Parâmetros:
    -----------
    model : estimator
        Modelo treinado (ex: PLS-DA)
    X : pd.DataFrame
        Dados espectrais originais (amostras x wavelengths)
    y_pred_original : array-like
        Predições originais do modelo
    spectral_cuts : list of tuples
        Lista de tuplas (nome_zona, inicio, fim) definindo regiões espectrais
    perturbation_value : float, default=0
        Valor a ser usado na perturbação (0 para zerar, 1 para mudar para 1, etc)
    metric : str, default='mean_abs_diff'
        Métrica para calcular a importância: 'mean_abs_diff', 'mean_diff', 'mean_relative_dev'.
        - 'mean_abs_diff': média da diferença absoluta
        - 'mean_diff': média da diferença (com sinal)
        - 'mean_relative_dev': média do desvio relativo (cuidado com divisão por zero)
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame com zona espectral e importância (diferença média nas predições)
    """
    import pandas as pd
    import numpy as np
    
    results = []
    
    for zone_name, start, end in spectral_cuts:
        # Criar cópia dos dados para perturbação
        X_perturbed = X.copy()
        # Identificar colunas dentro do range da zona espectral
        cols_to_perturb = [col for col in X.columns if start <= float(col) <= end]
        # Perturbar as colunas (mudar para o valor especificado)
        X_perturbed[cols_to_perturb] = perturbation_value
        # Fazer predição com dados perturbados
        y_pred_perturbed = model.predict(X_perturbed)
        # Calcular diferença entre predições
        if metric == 'mean_abs_diff':
            importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
        elif metric == 'mean_diff':
            importance = np.mean(y_pred_original - y_pred_perturbed)
        elif metric == 'mean_relative_dev':
            y_pred_original_safe = np.where(y_pred_original == 0, np.nan, y_pred_original)
            rel_dev = (y_pred_perturbed - y_pred_original) / y_pred_original_safe
            importance = np.nanmean(rel_dev)
        else:
            raise ValueError(f"Métrica '{metric}' não suportada. Use 'mean_abs_diff', 'mean_diff' ou 'mean_relative_dev'.")
        
        if metric == 'mean_relative_dev' or metric == 'mean_diff':
            pass  # importance já tem o sinal
            # Armazenar resultados
            results.append({
                'Zone': zone_name,
                'Start': start,
                'End': end,
                'Importance': importance,
                'Abs_Importance': np.abs(importance),
                'N_Features': len(cols_to_perturb)
            })
        else:
            # Armazenar resultados
            results.append({
                'Zone': zone_name,
                'Start': start,
                'End': end,
                'Importance': importance,
                'N_Features': len(cols_to_perturb)
            })

    # Criar DataFrame e ordenar por importância
    results_df = pd.DataFrame(results)
    if metric == 'mean_relative_dev' or metric == 'mean_diff':
        results_df = results_df.sort_values(by='Abs_Importance', ascending=False).reset_index(drop=True)
    else:
        results_df = results_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    return results_df

def _manual_block_permutation(
    estimator,
    X_eval: pd.DataFrame,
    y_eval: Union[pd.Series, np.ndarray],
    zone_cols: List[str],
    scoring: str,
    n_repeats: int,
    random_state: int,
    verbose: bool = False,
    classification_threshold: float = 0.5,
    task_type: str = None
) -> Tuple[float, float]:
    """
    Implementação manual de permutation importance para um bloco de colunas.
    
    Esta função é usada como fallback quando sklearn.permutation_importance
    não pode ser usada diretamente (ex: estimador incompatível com transformação).
    
    A estratégia é:
    1. Calcular o score baseline (sem permutação)
    2. Para cada repetição:
       a. Permutar as LINHAS do bloco de colunas (zone_cols) entre si
       b. Calcular o score com os dados permutados
    3. Retornar a média e desvio padrão da queda de score
    
    Suporta tanto tarefas de CLASSIFICAÇÃO quanto de REGRESSÃO.
    
    NOTA IMPORTANTE: Para tarefas de classificação com modelos que retornam 
    predições contínuas (como PLSRegression), as predições são convertidas 
    para classes usando um threshold (padrão=0.5).
    
    Parameters
    ----------
    estimator : sklearn estimator
        Modelo treinado com método predict() ou predict_proba()
    X_eval : pd.DataFrame
        Dados para avaliação (subconjunto do dataset original)
    y_eval : pd.Series or np.ndarray
        Rótulos verdadeiros correspondentes a X_eval.
        Para classificação: rótulos de classe
        Para regressão: valores numéricos contínuos
    zone_cols : list
        Lista de colunas que compõem a zona espectral a ser permutada
    scoring : str
        Nome da métrica de avaliação.
        Classificação: 'accuracy', 'f1', 'roc_auc', etc.
        Regressão: 'neg_root_mean_squared_error', 'neg_mean_squared_error',
                   'neg_mean_absolute_error', 'r2'
    n_repeats : int
        Número de repetições da permutação
    random_state : int
        Semente para reprodutibilidade
    verbose : bool
        Se True, imprime detalhes de cada repetição
    classification_threshold : float
        Threshold para converter predições contínuas em classes (padrão=0.5).
        Usado apenas para tarefas de classificação.
    task_type : str, optional
        Tipo de tarefa: 'classification' ou 'regression'.
        Se None, será inferido automaticamente a partir de y_eval e scoring.
    
    Returns
    -------
    tuple (float, float)
        (importância_média, desvio_padrão)
        onde importância = baseline_score - permuted_score
    
    Notes
    -----
    - Importância positiva: zona é importante (permutação piora o score)
    - Importância negativa: zona pode estar adicionando ruído
    - Importância zero: zona não afeta a predição
    - Para métricas negativas (neg_*), a importância é calculada corretamente
      considerando que piores scores são mais negativos.
    """
    rng = np.random.RandomState(random_state)
    
    # Converter y para array numpy se necessário
    if isinstance(y_eval, pd.Series):
        y_true = y_eval.values
    else:
        y_true = np.array(y_eval).flatten()
    
    # Inferir tipo de tarefa se não especificado
    if task_type is None:
        task_type = _infer_task_type(y_true, scoring)
    
    is_regression = (task_type == TASK_REGRESSION)
    
    if verbose:
        print(f"    Tipo de tarefa: {task_type}")
    
    # Configuração específica para classificação
    class_labels = ('A', 'B')  # Default
    is_pls_regression = False
    
    if not is_regression:
        # Detectar as classes únicas nos dados de verdade
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            # Assumimos que a primeira classe alfabeticamente é a "positiva" para threshold
            class_labels = tuple(sorted(unique_classes))
        else:
            class_labels = tuple(unique_classes) if len(unique_classes) > 0 else ('A', 'B')
        
        # Verificar se é um modelo de regressão (PLS) que retorna valores contínuos
        # PLSRegression não tem predict_proba, mas retorna valores contínuos
        is_pls_regression = hasattr(estimator, 'coef_') and not hasattr(estimator, 'predict_proba')
    
    # Mapeamento de scoring strings para funções de CLASSIFICAÇÃO
    classification_scoring_funcs = {
        'accuracy': lambda y_t, y_p: accuracy_score(y_t, y_p),
        'f1': lambda y_t, y_p: f1_score(y_t, y_p, average='binary', pos_label=class_labels[0]),
        'f1_weighted': lambda y_t, y_p: f1_score(y_t, y_p, average='weighted'),
        'f1_macro': lambda y_t, y_p: f1_score(y_t, y_p, average='macro'),
    }
    
    # Mapeamento de scoring strings para funções de REGRESSÃO
    # Nota: usamos as convenções do sklearn onde 'neg_*' significa que
    # o score é negado (maior = melhor, como outras métricas)
    regression_scoring_funcs = {
        'neg_root_mean_squared_error': lambda y_t, y_p: -np.sqrt(mean_squared_error(y_t, y_p)),
        'neg_mean_squared_error': lambda y_t, y_p: -mean_squared_error(y_t, y_p),
        'neg_mean_absolute_error': lambda y_t, y_p: -mean_absolute_error(y_t, y_p),
        'r2': lambda y_t, y_p: r2_score(y_t, y_p),
    }
    
    # Obter função de scoring baseada no tipo de tarefa
    needs_proba = False
    
    if is_regression:
        # Tarefa de regressão
        if scoring in regression_scoring_funcs:
            score_func = regression_scoring_funcs[scoring]
        else:
            warnings.warn(
                f"Scoring '{scoring}' não reconhecido para regressão. "
                f"Usando 'neg_root_mean_squared_error'. "
                f"Opções válidas: {list(regression_scoring_funcs.keys())}"
            )
            score_func = regression_scoring_funcs['neg_root_mean_squared_error']
    else:
        # Tarefa de classificação
        if scoring in classification_scoring_funcs:
            score_func = classification_scoring_funcs[scoring]
        elif scoring == 'roc_auc':
            needs_proba = True
            score_func = lambda y_t, y_p: roc_auc_score(y_t, y_p)
        else:
            warnings.warn(
                f"Scoring '{scoring}' não reconhecido para classificação. "
                f"Usando 'accuracy'. "
                f"Opções válidas: {list(classification_scoring_funcs.keys()) + ['roc_auc']}"
            )
            score_func = classification_scoring_funcs['accuracy']
    
    # Função auxiliar para obter predições de classe (apenas para classificação)
    def get_class_predictions(model, X_data):
        """Obtém predições de classe, tratando PLS e classificadores."""
        y_pred_raw = model.predict(X_data)
        
        # Se é PLS ou retorna valores contínuos, converter para classes
        if is_pls_regression or (isinstance(y_pred_raw, np.ndarray) and 
                                   y_pred_raw.dtype in [np.float64, np.float32]):
            # Verificar se são valores contínuos (não strings/objetos)
            y_flat = np.array(y_pred_raw).flatten()
            if np.issubdtype(y_flat.dtype, np.floating):
                return _convert_pls_prediction_to_class(
                    y_flat, 
                    threshold=classification_threshold,
                    class_labels=class_labels
                )
        
        # Se já são classes, retornar diretamente
        return np.array(y_pred_raw).flatten()
    
    # Função auxiliar para obter predições contínuas (para ROC AUC ou regressão)
    def get_continuous_predictions(model, X_data):
        """Obtém predições contínuas para métricas que requerem probabilidades ou regressão."""
        if hasattr(model, 'predict_proba') and not is_regression:
            return model.predict_proba(X_data)[:, 1]
        else:
            # PLS e modelos de regressão retornam valores contínuos diretamente
            return np.array(model.predict(X_data)).flatten()
    
    # Função auxiliar para obter predições de regressão
    def get_regression_predictions(model, X_data):
        """Obtém predições contínuas para tarefas de regressão."""
        y_pred_raw = model.predict(X_data)
        return np.array(y_pred_raw).flatten()
    
    # Calcular score baseline (sem permutação)
    X_original = X_eval.copy()
    
    try:
        if is_regression:
            # Tarefa de regressão: usar predições contínuas diretamente
            y_pred_baseline = get_regression_predictions(estimator, X_original)
            baseline_score = score_func(y_true, y_pred_baseline)
            
            if verbose:
                print(f"    Predições baseline (regressão): min={y_pred_baseline.min():.4f}, "
                      f"max={y_pred_baseline.max():.4f}, mean={y_pred_baseline.mean():.4f}")
                print(f"    y_true: min={y_true.min():.4f}, max={y_true.max():.4f}, "
                      f"mean={y_true.mean():.4f}")
        elif needs_proba:
            y_pred_baseline = get_continuous_predictions(estimator, X_original)
            # Para ROC AUC, converter labels para numérico
            y_numeric = np.where(y_true == class_labels[0], 1, 0)
            baseline_score = score_func(y_numeric, y_pred_baseline)
            
            if verbose:
                print(f"    Predições baseline: {np.unique(y_pred_baseline, return_counts=True)}")
                print(f"    y_true: {np.unique(y_true, return_counts=True)}")
        else:
            y_pred_baseline = get_class_predictions(estimator, X_original)
            baseline_score = score_func(y_true, y_pred_baseline)
            
            if verbose:
                print(f"    Predições baseline: {np.unique(y_pred_baseline, return_counts=True)}")
                print(f"    y_true: {np.unique(y_true, return_counts=True)}")
            
    except Exception as e:
        if verbose:
            print(f"    Erro no cálculo do baseline: {e}")
            import traceback
            traceback.print_exc()
        return 0.0, 0.0
    
    if verbose:
        print(f"    Baseline score: {baseline_score:.4f}")
    
    # Realizar permutações
    importance_scores = []
    
    for rep in range(n_repeats):
        # Copiar dados para não modificar o original
        X_permuted = X_eval.copy()
        
        # Permutar LINHAS do bloco de colunas (zone_cols)
        # Isso embaralha os valores espectrais entre as amostras
        n_samples = len(X_permuted)
        perm_indices = rng.permutation(n_samples)
        
        # Aplicar permutação às colunas da zona
        zone_values = X_permuted[zone_cols].values
        X_permuted[zone_cols] = zone_values[perm_indices]
        
        # Calcular score com dados permutados
        try:
            if is_regression:
                # Tarefa de regressão
                y_pred_perm = get_regression_predictions(estimator, X_permuted)
                permuted_score = score_func(y_true, y_pred_perm)
            elif needs_proba:
                y_pred_perm = get_continuous_predictions(estimator, X_permuted)
                y_numeric = np.where(y_true == class_labels[0], 1, 0)
                permuted_score = score_func(y_numeric, y_pred_perm)
            else:
                y_pred_perm = get_class_predictions(estimator, X_permuted)
                permuted_score = score_func(y_true, y_pred_perm)
        except Exception as e:
            if verbose:
                print(f"    Erro na repetição {rep+1}: {e}")
            permuted_score = baseline_score  # Assume sem impacto em caso de erro
        
        # Importância = baseline - permuted (queda de score)
        importance = baseline_score - permuted_score
        importance_scores.append(importance)
        
        if verbose and rep < 3:  # Mostrar apenas primeiras repetições para não poluir output
            print(f"    Rep {rep+1}: permuted_score={permuted_score:.4f}, importance={importance:.4f}")
    
    # Calcular média e desvio padrão
    importance_mean = np.mean(importance_scores)
    importance_std = np.std(importance_scores)
    
    return importance_mean, importance_std


def calculate_predicate_metrics_permutation(
    estimator,
    Xcalclass_prep: pd.DataFrame,
    y_calclass: pd.Series,
    folds_struct: Dict,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    scoring: str = 'accuracy',
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: bool = False,
    save_detailed_results: bool = True,
    task_type: str = None
) -> pd.DataFrame:
    """
    Calcula a importância de cada predicado usando Permutation Importance na zona espectral.
    
    Esta função é uma alternativa robusta a `calculate_predicate_metrics`. Em vez de
    usar métricas estatísticas (MI, Covariância), ela avalia o impacto REAL da zona
    espectral permutando os valores e medindo a queda de performance do modelo.
    
    Suporta tanto tarefas de CLASSIFICAÇÃO quanto de REGRESSÃO:
    - Classificação: usa métricas como accuracy, f1, roc_auc
    - Regressão: usa métricas como neg_root_mean_squared_error, neg_mean_squared_error, r2
    
    ESTRATÉGIA DE PERMUTAÇÃO POR BLOCO:
    -----------------------------------
    Para permutar a zona espectral inteira (e não coluna por coluna), usamos uma
    das seguintes estratégias:
    
    1. **Random Projection** (preferida): 
       - Cria uma projeção aleatória determinística do bloco em um único escalar
       - block_feature = X[zone_cols].values @ random_vector
       - Quando sklearn permuta esta coluna, está permutando linhas inteiras do bloco
       
    2. **Fallback Manual**:
       - Permuta diretamente as linhas do bloco de colunas
       - Calcula a métrica manualmente
       - Usado quando a estratégia 1 falha
    
    Parameters
    ----------
    estimator : sklearn estimator
        Modelo treinado (ex: PLSRegression para regressão, ou classificador)
        Deve ter método predict() ou predict_proba()
        
    Xcalclass_prep : pd.DataFrame
        Dataset de calibração pré-processado (n_samples × n_features)
        Colunas são energias/comprimentos de onda
        
    y_calclass : pd.Series
        Valores alvo correspondentes a Xcalclass_prep.
        Para classificação: rótulos de classe ('A', 'B', etc.)
        Para regressão: valores numéricos contínuos
        
    folds_struct : dict
        Estrutura de folds retornada por kfold_predicates_roundrobin():
        {
            'Fold_1': {
                'rule1': DataFrame(['Zone_Sum', 'Predicted_Y', 'Sample_Index']),
                'rule2': DataFrame([...]),
                ...
            },
            'Fold_2': {...},
            ...
        }
        
    predicates_df : pd.DataFrame
        DataFrame com informações dos predicados.
        Colunas obrigatórias: ['rule', 'zone', 'thresholds', 'operator']
        
    spectral_cuts : list of tuples
        Lista de cortes espectrais: [(nome, inicio, fim), ...]
        Ex: [('F1', 1.0, 100.0), ('F2', 200.0, 300.0), ...]
        
    scoring : str, default='accuracy'
        Métrica de scoring do sklearn.
        
        Para CLASSIFICAÇÃO:
        - 'accuracy': Acurácia (padrão)
        - 'f1': F1-score (binário)
        - 'f1_weighted': F1-score ponderado
        - 'f1_macro': F1-score macro
        - 'roc_auc': Área sob a curva ROC (requer predict_proba)
        
        Para REGRESSÃO:
        - 'neg_root_mean_squared_error': RMSE negativo (maior = melhor)
        - 'neg_mean_squared_error': MSE negativo (maior = melhor)
        - 'neg_mean_absolute_error': MAE negativo (maior = melhor)
        - 'r2': Coeficiente de determinação R²
        
    n_repeats : int, default=10
        Número de repetições da permutação (mais = mais estável)
        
    random_state : int, default=42
        Semente para reprodutibilidade
        
    n_jobs : int, default=1
        Número de jobs paralelos para sklearn (-1 = todos os cores)
        
    verbose : bool, default=False
        Se True, imprime detalhes do progresso
        
    save_detailed_results : bool, default=True
        Se True, salva resultados detalhados em atributo separado
        
    task_type : str, optional
        Tipo de tarefa: 'classification' ou 'regression'.
        Se None, será inferido automaticamente a partir de y_calclass e scoring.
        Use este parâmetro para forçar um tipo de tarefa específico.
    
    Returns
    -------
    pd.DataFrame
        DataFrame compatível com o retorno de calculate_predicate_metrics:
        {
            'Fold_1': DataFrame({
                'Predicate': ['rule1', 'rule2', ...],
                'Permutation': [0.45, 0.32, ...]
            }),
            ...
        }
        
        O valor da métrica é a importância por permutação (valor absoluto).
        
    Attributes (se save_detailed_results=True)
    ------------------------------------------
    A função também retorna um dicionário com resultados detalhados em:
    results_df.attrs['detailed_results'] contendo:
        - importance_mean: Importância média
        - importance_std: Desvio padrão
        - n_samples: Número de amostras no predicado
        - zone_columns: Colunas da zona espectral
        - baseline_score: Score sem permutação
    
    Notes
    -----
    - Predicados com n_samples <= 1 recebem importance = 0 (não há permutação possível)
    - Zonas espectrais vazias (0 colunas) também recebem importance = 0
    - O retorno é formatado para compatibilidade com o pipeline existente
    - Para métricas negativas (neg_*), a importância é calculada corretamente
    
    Example
    -------
    >>> # Exemplo de CLASSIFICAÇÃO
    >>> results = calculate_predicate_metrics_permutation(
    ...     estimator=pls_model,
    ...     Xcalclass_prep=Xcalclass_prep,
    ...     y_calclass=ycalclass,  # Labels: 'A', 'B'
    ...     folds_struct=folds_result,
    ...     predicates_df=predicates_quantiles[0],
    ...     spectral_cuts=spectral_cuts,
    ...     scoring='accuracy',
    ...     n_repeats=10,
    ...     verbose=True
    ... )
    >>> print(results['Fold_1'])
    
    >>> # Exemplo de REGRESSÃO
    >>> results_reg = calculate_predicate_metrics_permutation(
    ...     estimator=pls_regression_model,
    ...     Xcalclass_prep=Xcal_prep,
    ...     y_calclass=ycal,  # Valores contínuos: 1.5, 2.3, 0.8, ...
    ...     folds_struct=folds_result,
    ...     predicates_df=predicates_quantiles[0],
    ...     spectral_cuts=spectral_cuts,
    ...     scoring='neg_root_mean_squared_error',
    ...     n_repeats=10,
    ...     task_type='regression',
    ...     verbose=True
    ... )
    >>> print(results_reg['Fold_1'])
    """
    
    # =========================================================================
    # VALIDAÇÃO DE ENTRADAS
    # =========================================================================
    
    # Verificar se o estimator tem método predict
    if not hasattr(estimator, 'predict'):
        raise ValueError(
            "O estimator deve ter método predict(). "
            f"Tipo recebido: {type(estimator)}"
        )
    
    # Verificar estrutura de folds
    if not isinstance(folds_struct, dict):
        raise TypeError("folds_struct deve ser um dicionário")
    
    # Verificar colunas obrigatórias em predicates_df
    required_cols = ['rule', 'zone', 'thresholds', 'operator']
    missing_cols = [c for c in required_cols if c not in predicates_df.columns]
    if missing_cols:
        raise KeyError(f"Colunas faltando em predicates_df: {missing_cols}")
    
    # Verificar se y_calclass é Series ou converter
    if isinstance(y_calclass, np.ndarray):
        y_calclass = pd.Series(y_calclass)
    
    # =========================================================================
    # INICIALIZAÇÃO
    # =========================================================================
    
    rng = np.random.RandomState(random_state)
    
    # Inferir tipo de tarefa se não especificado
    if task_type is None:
        inferred_task_type = _infer_task_type(y_calclass, scoring)
    else:
        # Validar task_type fornecido
        if task_type not in [TASK_CLASSIFICATION, TASK_REGRESSION]:
            raise ValueError(
                f"task_type deve ser '{TASK_CLASSIFICATION}' ou '{TASK_REGRESSION}'. "
                f"Recebido: '{task_type}'"
            )
        inferred_task_type = task_type
    
    # Dicionário para resultados finais (compatível com calculate_predicate_metrics)
    metrics_results_dict = {}
    
    # Dicionário para resultados detalhados (permutation importance específicos)
    detailed_results = {}
    
    # Nome da métrica para compatibilidade (usamos 'Covariance' por padrão)
    # Isso mantém compatibilidade com o pipeline que espera essa coluna
    metric_name = 'Permutation'  # Mantém compatibilidade com pipeline existente
    
    total_folds = len(folds_struct)
    total_predicates_processed = 0
    total_predicates_skipped = 0
    
    if verbose:
        print("=" * 70)
        print("PERMUTATION IMPORTANCE PARA PREDICADOS")
        print("=" * 70)
        print(f"Tipo de tarefa: {inferred_task_type}")
        print(f"Métrica de scoring: {scoring}")
        print(f"Número de repetições: {n_repeats}")
        print(f"Random state: {random_state}")
        print(f"Total de folds: {total_folds}")
        print()
    
    # =========================================================================
    # LOOP PRINCIPAL: PROCESSAR CADA FOLD
    # =========================================================================
    
    for fold_idx, (fold_name, predicates_dict) in enumerate(folds_struct.items()):
        
        if verbose:
            print(f"\n[{fold_name}] Processando {len(predicates_dict)} predicados...")
        
        # Pular folds vazios
        if len(predicates_dict) == 0:
            if verbose:
                print(f"  VAZIO - pulando")
            metrics_results_dict[fold_name] = pd.DataFrame({
                'Predicate': [],
                metric_name: []
            })
            continue
        
        # Dicionário temporário para métricas deste fold
        fold_metrics = {}
        fold_detailed = {}
        
        # =====================================================================
        # LOOP: PROCESSAR CADA PREDICADO NO FOLD
        # =====================================================================
        
        for pred_rule, df_info in predicates_dict.items():
            
            total_predicates_processed += 1
            
            # -----------------------------------------------------------------
            # 1. OBTER ÍNDICES DE AMOSTRAS DO PREDICADO
            # -----------------------------------------------------------------
            
            # Os índices estão na coluna 'Sample_Index' do DataFrame do predicado
            sample_indices = df_info['Sample_Index'].values.tolist()
            n_samples = len(sample_indices)
            
            if verbose:
                print(f"  Predicado: {pred_rule} (n={n_samples})")
            
            # -----------------------------------------------------------------
            # 2. VERIFICAR CASOS LIMITES
            # -----------------------------------------------------------------
            
            # Caso: Predicado com poucas amostras (não é possível permutar)
            if n_samples <= 1:
                if verbose:
                    print(f"    SKIP: n_samples={n_samples} <= 1 (não é possível permutar)")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance_mean': 0.0,
                    'importance_std': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'n_samples <= 1'
                }
                total_predicates_skipped += 1
                continue
            
            # -----------------------------------------------------------------
            # 3. OBTER COLUNAS DA ZONA ESPECTRAL
            # -----------------------------------------------------------------
            
            try:
                zone_cols = get_zone_columns_from_predicate(
                    predicate_rule=pred_rule,
                    predicates_df=predicates_df,
                    spectral_cuts=spectral_cuts,
                    Xcal_columns=Xcalclass_prep.columns
                )
            except (KeyError, ValueError) as e:
                if verbose:
                    print(f"    ERRO ao obter colunas da zona: {e}")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance_mean': 0.0,
                    'importance_std': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': str(e)
                }
                total_predicates_skipped += 1
                continue
            
            # Verificar se a zona tem colunas
            if len(zone_cols) == 0:
                if verbose:
                    print(f"    SKIP: zona espectral vazia (0 colunas)")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance_mean': 0.0,
                    'importance_std': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'zona vazia'
                }
                total_predicates_skipped += 1
                continue
            
            if verbose:
                print(f"    Zona: {len(zone_cols)} colunas")
            
            # -----------------------------------------------------------------
            # 4. PREPARAR DADOS PARA AVALIAÇÃO
            # -----------------------------------------------------------------
            
            # Extrair subconjunto de dados para as amostras do predicado
            X_eval = Xcalclass_prep.iloc[sample_indices].copy()
            y_eval = y_calclass.iloc[sample_indices].copy()
            
            # -----------------------------------------------------------------
            # 5. ESTRATÉGIA DE PERMUTATION IMPORTANCE POR BLOCO
            # -----------------------------------------------------------------
            
            # ESTRATÉGIA A: Random Projection para criar coluna única representando o bloco
            # Isso permite usar sklearn.permutation_importance para permutar o bloco inteiro
            
            try:
                # Gerar vetor de projeção aleatória (determinístico via random_state)
                # Este vetor é fixo para cada zona, garantindo reprodutibilidade
                rp_vector = rng.randn(len(zone_cols))
                
                # Calcular projeção: cada linha vira um escalar
                # block_feature[i] = sum(X_eval[i, zone_cols] * rp_vector)
                block_feature = X_eval[zone_cols].values.dot(rp_vector)
                
                # Criar DataFrame modificado: remover colunas da zona, adicionar coluna única
                X_eval_modified = X_eval.drop(columns=zone_cols).copy()
                X_eval_modified['__zone_block__'] = block_feature
                
                # Verificar que o estimator pode trabalhar com este formato
                # Nota: O PLS original espera todas as colunas espectrais
                # Por isso, usaremos o fallback manual para a maioria dos casos
                
                # O sklearn.permutation_importance funciona bem apenas se o modelo
                # foi treinado com as mesmas features. Como estamos modificando as features,
                # vamos usar diretamente a abordagem de fallback manual que é mais robusta.
                
                raise NotImplementedError("Usando fallback manual por robustez")
                
            except Exception as e:
                # ESTRATÉGIA B (FALLBACK): Permutação manual do bloco
                # Esta é mais robusta porque mantém o formato original dos dados
                
                if verbose:
                    print(f"    Usando permutação manual do bloco")
                
                importance_mean, importance_std = _manual_block_permutation(
                    estimator=estimator,
                    X_eval=X_eval,
                    y_eval=y_eval,
                    zone_cols=zone_cols,
                    scoring=scoring,
                    n_repeats=n_repeats,
                    random_state=random_state + total_predicates_processed,  # Variar seed
                    verbose=verbose,
                    task_type=inferred_task_type
                )
            
            # -----------------------------------------------------------------
            # 6. ARMAZENAR RESULTADOS
            # -----------------------------------------------------------------
            
            fold_metrics[pred_rule] = np.abs(importance_mean)  # Valor absoluto para compatibilidade
            
            fold_detailed[pred_rule] = {
                'importance_mean': importance_mean,
                'importance_std': importance_std,
                'n_samples': n_samples,
                'zone_columns': zone_cols,
                'n_zone_features': len(zone_cols)
            }
            
            if verbose:
                print(f"    Importance: {importance_mean:.6f} ± {importance_std:.6f}")
        
        # =====================================================================
        # CONVERTER PARA DATAFRAME (compatível com calculate_predicate_metrics)
        # =====================================================================
        
        metrics_df = pd.DataFrame.from_dict(
            fold_metrics,
            orient='index',
            columns=[metric_name]
        )
        
        # Adicionar coluna de predicado
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Ordenar de forma DECRESCENTE (maiores valores = mais importantes)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Armazenar resultado
        metrics_results_dict[fold_name] = metrics_df
        detailed_results[fold_name] = fold_detailed
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    
    if verbose:
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        print(f"Folds processados: {total_folds}")
        print(f"Predicados processados: {total_predicates_processed}")
        print(f"Predicados ignorados (edge cases): {total_predicates_skipped}")
        print()
        for fold_name, df in metrics_results_dict.items():
            print(f"  {fold_name}: {len(df)} predicados com importance > 0")
    
    # =========================================================================
    # SALVAR RESULTADOS DETALHADOS (OPCIONAL)
    # =========================================================================
    
    # Criar DataFrame de resultados detalhados para análise posterior
    if save_detailed_results:
        detailed_rows = []
        for fold_name, fold_data in detailed_results.items():
            for pred_rule, pred_data in fold_data.items():
                detailed_rows.append({
                    'fold': fold_name,
                    'predicate': pred_rule,
                    'importance_mean': pred_data['importance_mean'],
                    'importance_std': pred_data['importance_std'],
                    'n_samples': pred_data['n_samples'],
                    'n_zone_features': pred_data.get('n_zone_features', 0),
                    'skip_reason': pred_data.get('skip_reason', None)
                })
        
        detailed_df = pd.DataFrame(detailed_rows)
        
        # Anexar como atributo do dicionário de resultados
        # Isso permite acesso sem quebrar o pipeline existente
        metrics_results_dict['__detailed_permutation_results__'] = detailed_df
    
    return metrics_results_dict


def calculate_predicate_perturbation(
    estimator,
    Xcalclass_prep: pd.DataFrame,
    folds_struct: Dict,
    predicates_df: pd.DataFrame,
    spectral_cuts: List[Tuple[str, float, float]],
    perturbation_value: float = 0,
    metric: str = 'mean_abs_diff',
    verbose: bool = False,
    save_detailed_results: bool = True
) -> Dict:
    """
    Calcula a importância de cada predicado usando Perturbação Espectral.
    
    Esta função é uma alternativa à permutação. Em vez de permutar valores,
    ela substitui os valores da zona espectral por um valor fixo (ex: 0) e
    mede o impacto na predição do modelo.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Modelo treinado com método predict()
    Xcalclass_prep : pd.DataFrame
        Dataset de calibração pré-processado (n_samples × n_features)
    folds_struct : dict
        Estrutura de folds no formato:
        {'Fold_1': {'rule1': DataFrame, 'rule2': DataFrame, ...}, ...}
    predicates_df : pd.DataFrame
        DataFrame com informações dos predicados (colunas: 'rule', 'zone', etc.)
    spectral_cuts : list of tuples
        Lista de cortes espectrais: [(nome, inicio, fim), ...]
    perturbation_value : float, default=0
        Valor usado para perturbar a zona (0 = zerar a zona)
    metric : str, default='mean_abs_diff'
        Métrica: 'mean_abs_diff', 'mean_diff' ou 'mean_relative_dev'
    verbose : bool, default=False
        Se True, imprime detalhes do progresso
    save_detailed_results : bool, default=True
        Se True, salva resultados detalhados
    
    Returns
    -------
    dict
        Dicionário no formato compatível com calculate_predicate_metrics_permutation:
        {'Fold_1': DataFrame({'Predicate': [...], 'Perturbation': [...]}), ...}
    
    Example
    -------
    >>> results = calculate_predicate_perturbation(
    ...     estimator=pls_model,
    ...     Xcalclass_prep=Xcal_prep,
    ...     folds_struct=folds_result,
    ...     predicates_df=predicates_quantiles[0],
    ...     spectral_cuts=spectral_cuts,
    ...     perturbation_value=0,
    ...     metric='mean_abs_diff',
    ...     verbose=True
    ... )
    >>> print(results['Fold_1'])
    """
    
    # =========================================================================
    # VALIDAÇÃO DE ENTRADAS
    # =========================================================================
    
    # Verificar se o estimator tem método predict
    if not hasattr(estimator, 'predict'):
        # Lança erro se o modelo não tiver método predict
        raise ValueError(f"O estimator deve ter método predict(). Tipo: {type(estimator)}")
    
    # Verificar se folds_struct é dicionário
    if not isinstance(folds_struct, dict):
        # Lança erro se a estrutura de folds não for dicionário
        raise TypeError("folds_struct deve ser um dicionário")
    
    # Verificar colunas obrigatórias em predicates_df
    required_cols = ['rule', 'zone']  # Colunas mínimas necessárias
    missing_cols = [c for c in required_cols if c not in predicates_df.columns]
    if missing_cols:
        # Lança erro se faltar alguma coluna obrigatória
        raise KeyError(f"Colunas faltando em predicates_df: {missing_cols}")
    
    # =========================================================================
    # INICIALIZAÇÃO
    # =========================================================================
    
    # Dicionário para armazenar resultados finais (compatível com pipeline existente)
    metrics_results_dict = {}
    
    # Dicionário para armazenar resultados detalhados
    detailed_results = {}
    
    # Nome da coluna de métrica no DataFrame de saída
    metric_name = 'Perturbation'
    
    # Contadores para estatísticas
    total_folds = len(folds_struct)  # Total de folds a processar
    total_predicates_processed = 0   # Contador de predicados processados
    total_predicates_skipped = 0     # Contador de predicados ignorados
    
    # Log inicial se verbose
    if verbose:
        print("=" * 70)
        print("PERTURBATION IMPORTANCE PARA PREDICADOS")
        print("=" * 70)
        print(f"Valor de perturbação: {perturbation_value}")
        print(f"Métrica: {metric}")
        print(f"Total de folds: {total_folds}")
        print()
    
    # =========================================================================
    # LOOP PRINCIPAL: PROCESSAR CADA FOLD
    # =========================================================================
    
    # Iterar sobre cada fold na estrutura
    for fold_idx, (fold_name, predicates_dict) in enumerate(folds_struct.items()):
        
        # Log do fold atual
        if verbose:
            print(f"\n[{fold_name}] Processando {len(predicates_dict)} predicados...")
        
        # Verificar se o fold está vazio
        if len(predicates_dict) == 0:
            # Se vazio, criar DataFrame vazio e pular para próximo fold
            if verbose:
                print(f"  VAZIO - pulando")
            metrics_results_dict[fold_name] = pd.DataFrame({
                'Predicate': [],
                metric_name: []
            })
            continue
        
        # Dicionário temporário para métricas deste fold
        fold_metrics = {}
        
        # Dicionário temporário para resultados detalhados deste fold
        fold_detailed = {}
        
        # =====================================================================
        # LOOP: PROCESSAR CADA PREDICADO NO FOLD
        # =====================================================================
        
        # Iterar sobre cada predicado do fold
        for pred_rule, df_info in predicates_dict.items():
            
            # Incrementar contador de predicados processados
            total_predicates_processed += 1
            
            # -----------------------------------------------------------------
            # 1. OBTER ÍNDICES DE AMOSTRAS DO PREDICADO
            # -----------------------------------------------------------------
            
            # Extrair índices das amostras que pertencem a este predicado
            sample_indices = df_info['Sample_Index'].values.tolist()
            
            # Número de amostras no predicado
            n_samples = len(sample_indices)
            
            # Log do predicado atual
            if verbose:
                print(f"  Predicado: {pred_rule} (n={n_samples})")
            
            # -----------------------------------------------------------------
            # 2. VERIFICAR CASOS LIMITES
            # -----------------------------------------------------------------
            
            # Se não há amostras, não é possível calcular importância
            if n_samples == 0:
                if verbose:
                    print(f"    SKIP: n_samples=0 (sem amostras)")
                # Atribuir importância zero
                fold_metrics[pred_rule] = 0.0
                # Salvar detalhes
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'n_samples = 0'
                }
                # Incrementar contador de skips
                total_predicates_skipped += 1
                continue
            
            # -----------------------------------------------------------------
            # 3. OBTER INFORMAÇÕES DA ZONA ESPECTRAL
            # -----------------------------------------------------------------
            
            # Tentar obter colunas da zona espectral do predicado
            try:
                # Usar função auxiliar para obter colunas da zona
                zone_cols = get_zone_columns_from_predicate(
                    predicate_rule=pred_rule,
                    predicates_df=predicates_df,
                    spectral_cuts=spectral_cuts,
                    Xcal_columns=Xcalclass_prep.columns
                )
            except (KeyError, ValueError) as e:
                # Se erro ao obter zona, atribuir importância zero
                if verbose:
                    print(f"    ERRO ao obter zona: {e}")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': str(e)
                }
                total_predicates_skipped += 1
                continue
            
            # Verificar se a zona tem colunas
            if len(zone_cols) == 0:
                # Se zona vazia, atribuir importância zero
                if verbose:
                    print(f"    SKIP: zona espectral vazia")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': [],
                    'skip_reason': 'zona vazia'
                }
                total_predicates_skipped += 1
                continue
            
            # Log das colunas da zona
            if verbose:
                print(f"    Zona: {len(zone_cols)} colunas")
            
            # -----------------------------------------------------------------
            # 4. OBTER LIMITES DA ZONA PARA PERTURBAÇÃO
            # -----------------------------------------------------------------
            
            # Encontrar nome da zona associada ao predicado
            mask_pred = predicates_df['rule'] == pred_rule
            zone_name = predicates_df.loc[mask_pred, 'zone'].values[0]
            
            # Encontrar limites (start, end) da zona nos spectral_cuts
            zone_start, zone_end = None, None
            for cut in spectral_cuts:
                # Extrair nome e limites do cut
                if len(cut) == 3:
                    name, start, end = cut
                elif len(cut) == 2:
                    start, end = cut
                    name = f"{start}-{end}"
                else:
                    continue
                
                # Verificar se é a zona correta
                if name == zone_name:
                    zone_start, zone_end = float(start), float(end)
                    break
            
            # Se não encontrou limites, pular
            if zone_start is None or zone_end is None:
                if verbose:
                    print(f"    SKIP: limites da zona não encontrados")
                fold_metrics[pred_rule] = 0.0
                fold_detailed[pred_rule] = {
                    'importance': 0.0,
                    'n_samples': n_samples,
                    'zone_columns': zone_cols,
                    'skip_reason': 'limites não encontrados'
                }
                total_predicates_skipped += 1
                continue
            
            # -----------------------------------------------------------------
            # 5. EXTRAIR DADOS DAS AMOSTRAS DO PREDICADO
            # -----------------------------------------------------------------
            
            # Extrair subconjunto de dados para as amostras do predicado
            X_eval = Xcalclass_prep.iloc[sample_indices].copy()
            
            # -----------------------------------------------------------------
            # 6. CALCULAR PREDIÇÃO ORIGINAL (SEM PERTURBAÇÃO)
            # -----------------------------------------------------------------
            
            # Fazer predição com dados originais
            y_pred_original = estimator.predict(X_eval)
            
            # Achatar array se necessário
            y_pred_original = np.array(y_pred_original).flatten()
            
            # -----------------------------------------------------------------
            # 7. PERTURBAR ZONA ESPECTRAL E CALCULAR NOVA PREDIÇÃO
            # -----------------------------------------------------------------
            
            # Criar cópia dos dados para perturbação
            X_perturbed = X_eval.copy()
            
            # Substituir valores da zona pelo valor de perturbação
            X_perturbed[zone_cols] = perturbation_value
            
            # Fazer predição com dados perturbados
            y_pred_perturbed = estimator.predict(X_perturbed)
            
            # Achatar array se necessário
            y_pred_perturbed = np.array(y_pred_perturbed).flatten()
            
            # -----------------------------------------------------------------
            # 8. CALCULAR IMPORTÂNCIA BASEADA NA MÉTRICA ESCOLHIDA
            # -----------------------------------------------------------------
            
            # Calcular importância de acordo com a métrica
            if metric == 'mean_abs_diff':
                # Média da diferença absoluta entre predições
                importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
            elif metric == 'mean_diff':
                # Média da diferença (com sinal)
                importance = np.mean(y_pred_original - y_pred_perturbed)
            elif metric == 'mean_relative_dev':
                # Média do desvio relativo (cuidado com divisão por zero)
                y_safe = np.where(y_pred_original == 0, np.nan, y_pred_original)
                rel_dev = (y_pred_perturbed - y_pred_original) / y_safe
                importance = np.nanmean(rel_dev)
            else:
                # Métrica não reconhecida, usar mean_abs_diff como fallback
                if verbose:
                    print(f"    AVISO: métrica '{metric}' não reconhecida, usando mean_abs_diff")
                importance = np.mean(np.abs(y_pred_original - y_pred_perturbed))
            
            # -----------------------------------------------------------------
            # 9. ARMAZENAR RESULTADOS
            # -----------------------------------------------------------------
            
            # Para ranking, usar valor absoluto para métricas com sinal
            if metric in ['mean_diff', 'mean_relative_dev']:
                # Usar valor absoluto para ordenação
                fold_metrics[pred_rule] = np.abs(importance)
            else:
                # mean_abs_diff já é absoluto
                fold_metrics[pred_rule] = importance
            
            # Salvar detalhes completos
            fold_detailed[pred_rule] = {
                'importance': importance,
                'importance_abs': np.abs(importance),
                'n_samples': n_samples,
                'zone_columns': zone_cols,
                'n_zone_features': len(zone_cols),
                'zone_name': zone_name,
                'zone_start': zone_start,
                'zone_end': zone_end
            }
            
            # Log da importância calculada
            if verbose:
                print(f"    Importance: {importance:.6f}")
        
        # =====================================================================
        # CONVERTER PARA DATAFRAME (compatível com pipeline existente)
        # =====================================================================
        
        # Criar DataFrame a partir do dicionário de métricas
        metrics_df = pd.DataFrame.from_dict(
            fold_metrics,
            orient='index',
            columns=[metric_name]
        )
        
        # Adicionar coluna de predicado
        metrics_df.insert(0, 'Predicate', metrics_df.index)
        
        # Resetar índice para ter índice numérico
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Ordenar de forma DECRESCENTE (maiores valores = mais importantes)
        metrics_df = metrics_df.sort_values(by=metric_name, ascending=False)
        
        # Resetar índice após ordenação
        metrics_df = metrics_df.reset_index(drop=True)
        
        # Armazenar resultado do fold
        metrics_results_dict[fold_name] = metrics_df
        
        # Armazenar resultados detalhados do fold
        detailed_results[fold_name] = fold_detailed
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    
    # Imprimir resumo se verbose
    if verbose:
        print("\n" + "=" * 70)
        print("RESUMO")
        print("=" * 70)
        print(f"Folds processados: {total_folds}")
        print(f"Predicados processados: {total_predicates_processed}")
        print(f"Predicados ignorados: {total_predicates_skipped}")
        print()
        # Mostrar resumo por fold
        for fold_name, df in metrics_results_dict.items():
            # Ignorar chave especial de resultados detalhados
            if fold_name.startswith('__'):
                continue
            print(f"  {fold_name}: {len(df)} predicados")
    
    # =========================================================================
    # SALVAR RESULTADOS DETALHADOS (OPCIONAL)
    # =========================================================================
    
    # Se solicitado, criar DataFrame com todos os detalhes
    if save_detailed_results:
        # Lista para armazenar linhas do DataFrame detalhado
        detailed_rows = []
        
        # Iterar sobre folds e predicados
        for fold_name, fold_data in detailed_results.items():
            for pred_rule, pred_data in fold_data.items():
                # Adicionar linha com informações do predicado
                detailed_rows.append({
                    'fold': fold_name,
                    'predicate': pred_rule,
                    'importance': pred_data['importance'],
                    'importance_abs': pred_data.get('importance_abs', np.abs(pred_data['importance'])),
                    'n_samples': pred_data['n_samples'],
                    'n_zone_features': pred_data.get('n_zone_features', 0),
                    'zone_name': pred_data.get('zone_name', None),
                    'skip_reason': pred_data.get('skip_reason', None)
                })
        
        # Criar DataFrame de resultados detalhados
        detailed_df = pd.DataFrame(detailed_rows)
        
        # Anexar como chave especial no dicionário de resultados
        metrics_results_dict['__detailed_perturbation_results__'] = detailed_df
    
    # Retornar dicionário com resultados
    return metrics_results_dict
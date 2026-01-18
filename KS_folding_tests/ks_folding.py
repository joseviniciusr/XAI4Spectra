# Amostragem Cumulativa via Kennard-Stone + Round-Robin
# Abordagem simplificada para criar folds determinísticos e balanceados
# 
# DUAS ESTRATÉGIAS DISPONÍVEIS:
# 1. GLOBAL (per_predicate=False): KS aplicado globalmente, folds compartilhados entre predicados
# 2. INDIVIDUAL (per_predicate=True): KS aplicado por predicado, folds independentes

import numpy as np
import pandas as pd
import kennard_stone as ks


def ks_ordered_indices(X):
    """
    Obtém a ordem de seleção das amostras via Kennard-Stone.
    Retorna índices ordenados do mais representativo ao menos representativo.
    
    Parâmetros:
    -----------
    - X : pd.DataFrame - Dados com features (n_samples × n_features)
    
    Retorna:
    --------
    - list - Índices ordenados por representatividade (KS)
    """
    n = len(X)
    if n <= 2:
        return list(range(n))
    
    # Obtém índices ordenados por KS (calibração contém amostras mais representativas primeiro)
    X_cal, X_test = ks.train_test_split(X, test_size=0.01)  # quase todas vão para calibração
    
    # Os índices de calibração já estão na ordem de seleção KS
    cal_indices = X_cal.index.tolist()
    test_indices = X_test.index.tolist()
    
    # Retorna todos os índices: calibração (ordenados por KS) + teste (restantes)
    return cal_indices + test_indices


def ks_ordered_indices_1d(values, original_indices=None):
    """
    Aplica Kennard-Stone em dados unidimensionais (um único predicado/zona) criando
    um DataFrame 2D com uma coluna adicional de zeros.

    RAZÃO: Algumas implementações de KS esperam entradas 2D. Adicionar uma coluna
    constante com zeros torna a matriz 2D válida para o algoritmo sem alterar
    as distâncias euclidianas entre amostras, pois a diferença entre zeros é zero.

    Parâmetros:
    -----------
    - values : array-like (pd.Series, list, np.ndarray) - Valores unidimensionais
      (ex: valores agregados da zona) para as amostras do predicado
    - original_indices : array-like, opcional - Índices originais das amostras no
      dataset completo (se None, usa np.arange(n))

    Retorna:
    --------
    - list - Índices originais ordenados por representatividade (KS)
    """
    import numpy as np
    import pandas as pd

    # Converte entrada para Series para trabalhar de forma consistente
    vals = pd.Series(values).reset_index(drop=True)

    n = len(vals)

    # Se não vierem índices originais, usamos 0..n-1
    if original_indices is None:
        original_indices = np.arange(n)
    original_indices = np.array(original_indices)

    # Casos triviais: poucas amostras → não aplicar KS, retorna índices originais
    if n <= 2:
        return original_indices.tolist()

    # Construção do DataFrame 2D para KS: coluna de valor + coluna de zeros
    # - 'zone_value' : float (mantém a variação do predicado)
    # - 'zero_col'   : zeros (float) — garante 2D sem alterar distâncias
    X_2d = pd.DataFrame({
        'zone_value': vals.astype(float),      # força float para evitar issues numéricas
        'zero_col': np.zeros(n, dtype=float)   # coluna constante com zeros
    })

    # Aplica KS para obter a ordem local (usa a função já definida ks_ordered_indices)
    local_ordered = ks_ordered_indices(X_2d)

    # Converte índices locais (0..n-1) para índices do dataset original
    global_ordered = [original_indices[i] for i in local_ordered]

    return global_ordered

def ks_roundrobin_kfold(X, k_folds=10):
    """
    Cria k folds usando ordenação Kennard-Stone + distribuição round-robin.
    
    Lógica:
    1. Ordena todas as amostras via KS (mais representativas primeiro)
    2. Distribui em round-robin: amostra 1→fold 1, amostra 2→fold 2, amostra k→fold k, amostra k+1→fold 1, amostra k+2→fold 2, etc.
    
    Isso garante que cada fold tenha diversidade similar.
    
    Parâmetros:
    -----------
    - X : pd.DataFrame - Dados (n_samples × n_features)
    - k_folds : int - Número de folds
    
    Retorna:
    --------
    - list[list] - Lista de listas com índices de amostras por fold
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    n_samples = len(X)
    if k_folds < 2 or k_folds > n_samples:
        raise ValueError(f'k_folds deve estar entre 2 e {n_samples}')
    
    # Obtém índices ordenados por KS
    ordered_indices = ks_ordered_indices(X)
    
    # Distribui round-robin para os folds
    folds = [[] for _ in range(k_folds)] # o _ é usado quando a variável não é necessária, enquanto o [] cria uma lista vazia para cada fold
    for i, idx in enumerate(ordered_indices): # 
        fold_num = i % k_folds # Ex: se k_folds=3, i=0 gera fold_num=0, i=1 gera um fold_num=1, i=2 fold_num=2, i=3 fold_num=0, i=4 será 1, etc.  Ou seja, os fold_num serão 0,1,2,0,1,2,...
        folds[fold_num].append(idx) # distribui o índice idx para o fold correspondente
    
    return folds


def ks_roundrobin_kfold_1d(values, original_indices, k_folds=10):
    """
    Cria k folds usando KS + round-robin para dados unidimensionais.
    
    Esta função é usada na abordagem PER-PREDICATE, onde cada predicado
    tem seus próprios folds baseados apenas nas amostras que o satisfazem.
    
    Parâmetros:
    -----------
    - values : np.array - Valores unidimensionais (ex: valores agregados da zona)
    - original_indices : list - Índices originais das amostras
    - k_folds : int - Número de folds
    
    Retorna:
    --------
    - list[list] - Lista de listas com índices originais por fold
    """
    n_samples = len(values)
    
    # Ajusta k_folds se necessário
    k_folds = min(k_folds, n_samples)
    if k_folds < 2:
        # Poucas amostras: retorna todas em um único fold
        return [list(original_indices)]
    
    # Obtém índices ordenados por KS (usando solução 2D)
    ordered_indices = ks_ordered_indices_1d(values, original_indices)
    
    # Distribui round-robin para os folds
    folds = [[] for _ in range(k_folds)]
    for i, idx in enumerate(ordered_indices):
        fold_num = i % k_folds
        folds[fold_num].append(idx)
    
    return folds

def kfold_predicates_roundrobin(zone_sums_df, y_predicted_numeric, predicates_df,
                                 k_folds=10, min_samples_ratio=0.20, verbose=True,
                                 per_predicate=False):
    """
    Cria folds via KS + round-robin e processa predicados por fold.
    
    DUAS ESTRATÉGIAS DISPONÍVEIS:
    
    1. GLOBAL (per_predicate=False):
       - KS é aplicado globalmente em TODAS as amostras
       - Todos os predicados compartilham exatamente os mesmos folds
       - Vantagem: Consistência total entre predicados (mesmas amostras nos mesmos folds)
       - Desvantagem: Predicados com baixa cobertura podem ser eliminados
    
    2. INDIVIDUAL (per_predicate=True):
       - KS é aplicado individualmente para cada predicado
       - Cada predicado tem seus próprios folds independentes
       - Vantagem: Maximiza representatividade dentro de cada predicado
       - Vantagem: Menos predicados eliminados por min_samples
       - Desvantagem: Folds inconsistentes entre predicados
       - Desvantagem: Maior custo computacional
    
    Parâmetros:
    -----------
    - zone_sums_df : pd.DataFrame - DataFrame (n_samples × zones) com valores agregados
    - y_predicted_numeric : pd.Series - Predições contínuas do PLS-DA
    - predicates_df : pd.DataFrame - DataFrame com colunas ['rule','zone','thresholds','operator']
    - k_folds : int - Número de folds
    - min_samples_ratio : float - Proporção mínima de amostras por predicado (default=0.20)
    - verbose : bool - Se True, imprime predicados eliminados
    - per_predicate : bool - Se True, aplica KS individualmente por predicado (NOVA ESTRATÉGIA)
    
    Retorna:
    --------
    - dict : {'Fold_1': {rule: df, ...}, 'Fold_2': {...}, ...}
    """
    if not isinstance(zone_sums_df, pd.DataFrame):
        zone_sums_df = pd.DataFrame(zone_sums_df)
    if not isinstance(y_predicted_numeric, pd.Series):
        y_predicted_numeric = pd.Series(y_predicted_numeric)
    
    n_total = len(zone_sums_df)
    
    # Calcula número mínimo de amostras por predicado baseado no tamanho do fold
    samples_per_fold = n_total // k_folds # o // é divisão inteira que retorna apenas a parte inteira do resultado
    min_samples = max(2, int(samples_per_fold * min_samples_ratio))
    
    if verbose:
        print(f"Configuração KS + Round-Robin")
        print(f"Estratégia: {'PER-PREDICATE (individual)' if per_predicate else 'GLOBAL (compartilhada)'}")
        print(f"Total de amostras: {n_total}")
        print(f"Número de folds: {k_folds}")
        print(f"Amostras por fold (aprox.): {samples_per_fold}")
        print(f"Mínimo de amostras por predicado: {min_samples} ({min_samples_ratio*100:.0f}% do fold)")
        print()
    
    # ESTRATÉGIA 1: GLOBAL (per_predicate=False)
    # KS aplicado globalmente, folds compartilhados
    if not per_predicate:
        # Cria folds via KS + round-robin
        # Aplica Kennard-Stone diretamente em TODAS as amostras (sem separar por classe)
        # Isso ordena as amostras por representatividade e distribui ciclicamente
        fold_indices = ks_roundrobin_kfold(zone_sums_df, k_folds=k_folds)
        
        # Inicializa dicionário vazio para armazenar os resultados finais
        # Estrutura: {'Fold_1': {rule1: df1, rule2: df2, ...}, 'Fold_2': {...}, ...}
        folds_dict = {}
        
        # Set (conjunto) para rastrear quais predicados foram eliminados
        # Usado para evitar imprimir avisos duplicados quando um predicado falha em múltiplos folds
        eliminated_predicates = set()
        
        # Loop sobre cada fold criado pelo KS + round-robin
        for fold_num, fold_idx in enumerate(fold_indices, start=1):
            # fold_num: número do fold (1, 2, 3, ...)
            # fold_idx: lista de índices das amostras neste fold
            
            # Cria nome do fold (ex: 'Fold_1', 'Fold_2', ...)
            fold_name = f'Fold_{fold_num}'
            
            # Dicionário para armazenar os predicados válidos deste fold
            # Estrutura: {rule1: df1, rule2: df2, ...}
            pred_dict = {}
            
            # Loop sobre cada predicado (regra) definido em predicates_df
            for _, row in predicates_df.iterrows():
                # Extrai informações do predicado da linha atual
                rule = row['rule']           # ex: 'F1 <= 10.5'
                zone = row['zone']           # ex: 'F1'
                thr = float(row['thresholds'])  # ex: 10.5
                op = row['operator']         # ex: '<='
                
                # Obtém os valores da zona para as amostras deste fold
                # loc[fold_idx, zone] seleciona linhas em fold_idx e coluna 'zone'
                zvals = zone_sums_df.loc[fold_idx, zone].values
                
                # Aplica o operador do predicado para criar máscara booleana
                if op == '<=':
                    mask = zvals <= thr  # True onde valor <= threshold
                elif op == '>':
                    mask = zvals > thr   # True onde valor > threshold
                else:
                    continue  # Operador desconhecido, pula este predicado
                
                # Filtra apenas os índices das amostras que satisfazem o predicado
                # np.array(fold_idx)[mask] aplica a máscara booleana na lista de índices
                satisfied_idx = np.array(fold_idx)[mask]
                
                # Verifica se o predicado tem cobertura mínima neste fold
                if len(satisfied_idx) < min_samples:
                    # Predicado não tem amostras suficientes neste fold
                    
                    # Imprime aviso apenas se ainda não foi eliminado (evita duplicatas)
                    if rule not in eliminated_predicates and verbose:
                        print(f"[{fold_name}] Predicado eliminado: '{rule}' "
                              f"(apenas {len(satisfied_idx)} amostras, mínimo={min_samples})")
                        # Adiciona ao set de eliminados para não imprimir novamente
                        eliminated_predicates.add(rule)
                    
                    # Pula para o próximo predicado
                    continue
                
                # Cria DataFrame com informações das amostras que satisfazem
                pred_dict[rule] = pd.DataFrame({
                    'Zone_Sum': zone_sums_df.loc[satisfied_idx, zone].values,      # Valores da zona
                    'Predicted_Y': y_predicted_numeric.iloc[satisfied_idx].values, # Predições PLS-DA
                    'Sample_Index': satisfied_idx                                   # Índices originais
                })
            
            # Armazena o dicionário de predicados válidos deste fold no resultado final
            folds_dict[fold_name] = pred_dict
        
        # Imprime resumo final da estratégia GLOBAL (se verbose=True)
        if verbose:
            print()  # Linha em branco para separação visual
            print(f"=== Resumo (Estratégia GLOBAL - SEM estratificação) ===")
            print(f"Predicados totais: {len(predicates_df)}")  # Total de predicados testados
            print(f"Predicados eliminados (em pelo menos 1 fold): {len(eliminated_predicates)}")  # Quantos foram descartados
            print(f"Folds criados: {len(folds_dict)}")  # Número de folds (deve ser k_folds)
            
            # Loop para imprimir quantos predicados válidos há em cada fold
            for fname, pdict in folds_dict.items():
                print(f"  {fname}: {len(pdict)} predicados válidos")
        
        # Retorna o dicionário final com todos os folds e seus predicados
        return folds_dict
    
    # ESTRATÉGIA 2: PER-PREDICATE (per_predicate=True)
    # KS aplicado individualmente por predicado, folds independentes
    else:
        if verbose:
            print("Iniciando estratégia PER-PREDICATE...")
            print("KS será aplicado individualmente para cada predicado.\n")
        
        # Estrutura para armazenar folds por predicado
        # predicate_folds[rule] = {fold_num: [indices]}
        predicate_folds = {}
        
        eliminated_predicates = set() # Para predicados eliminados
        predicate_stats = {}  # Para estatísticas
        
        # FASE 1: Aplica KS individualmente para cada predicado
        for _, row in predicates_df.iterrows(): # o iterrows() itera sobre as linhas do DataFrame e retorna um índice e uma Series com os dados da linha
            rule = row['rule']
            zone = row['zone']
            thr = float(row['thresholds'])
            op = row['operator']
            
            # Identifica amostras que satisfazem o predicado (globalmente)
            zvals_global = zone_sums_df[zone].values
            
            if op == '<=':
                mask_global = zvals_global <= thr
            elif op == '>':
                mask_global = zvals_global > thr
            else:
                continue
            
            # Índices das amostras que satisfazem o predicado
            satisfied_indices = np.where(mask_global)[0].tolist() # o np.where retorna uma tupla onde o primeiro elemento é um array com os índices que satisfazem a condição
            n_satisfied = len(satisfied_indices) # número de amostras que satisfazem o predicado
            
            # Verifica se há amostras suficientes para criar folds
            if n_satisfied < k_folds: # se o número de amostras que satisfazem o predicado for menor que o número de folds
                if verbose:
                    print(f"[ELIMINADO] '{rule}': apenas {n_satisfied} amostras "
                          f"(mínimo para {k_folds} folds)")
                eliminated_predicates.add(rule)
                continue
            
            # Valores da zona apenas para as amostras que satisfazem
            zvals_satisfied = zone_sums_df.loc[satisfied_indices, zone].values
            
            # APLICA KS + ROUND-ROBIN APENAS NAS AMOSTRAS DO PREDICADO
            # Usa a função especializada para dados 1D
            pred_folds = ks_roundrobin_kfold_1d(
                values=zvals_satisfied,
                original_indices=satisfied_indices,
                k_folds=k_folds
            )
            
            # Verifica min_samples em cada fold
            valid_folds = True
            for fold_idx_list in pred_folds: # fold_idx_list são os índices das amostras que satisfazem o predicado naquele fold
                if len(fold_idx_list) < min_samples: # se o número de amostras naquele fold for menor que o mínimo
                    valid_folds = False # seta como inválido
                    break # sai do loop quando encontrar o primeiro fold inválido
            
            if not valid_folds: # se algum fold for inválido
                if verbose:
                    fold_sizes = [len(f) for f in pred_folds]
                    print(f"[ELIMINADO] '{rule}': fold com menos de {min_samples} amostras "
                          f"(tamanhos: {fold_sizes})")
                eliminated_predicates.add(rule)
                continue
            
            # Armazena os folds do predicado
            predicate_folds[rule] = pred_folds
            
            # Estatísticas
            predicate_stats[rule] = {
                'n_total': n_satisfied,
                'fold_sizes': [len(f) for f in pred_folds],
                'zone': zone
            }
        
        # FASE 2: Reorganiza em estrutura {Fold_N: {rule: df}}
        folds_dict = {f'Fold_{i+1}': {} for i in range(k_folds)} # cria dicionário com chaves Fold_1, Fold_2, ..., Fold_k
        
        for rule, pred_folds in predicate_folds.items(): # itera sobre os predicados válidos, o item retorna uma tupla (chave, valor)
            zone = predicate_stats[rule]['zone'] # obtém a zona associada ao predicado
            
            for fold_num, fold_indices in enumerate(pred_folds): # itera sobre os folds do predicado
                fold_name = f'Fold_{fold_num + 1}' # nome do fold atual
                
                # Cria DataFrame com informações do predicado neste fold
                folds_dict[fold_name][rule] = pd.DataFrame({
                    'Zone_Sum': zone_sums_df.loc[fold_indices, zone].values,
                    'Predicted_Y': y_predicted_numeric.iloc[fold_indices].values,
                    'Sample_Index': fold_indices
                })
        
        # FASE 3: Resumo e estatísticas
        if verbose:
            print()
            print(f"Resumo (Estratégia PER-PREDICATE)")
            print(f"Predicados totais: {len(predicates_df)}")
            print(f"Predicados válidos: {len(predicate_folds)}")
            print(f"Predicados eliminados: {len(eliminated_predicates)}")
            print(f"Folds criados: {len(folds_dict)}")
            print()
            print("Estatísticas por predicado válido:")
            for rule, stats in list(predicate_stats.items())[:5]:  # Mostra apenas top 5
                print(f"  '{rule}': {stats['n_total']} amostras, "
                      f"folds: {stats['fold_sizes']}")
            if len(predicate_stats) > 5:
                print(f"  ... e mais {len(predicate_stats) - 5} predicados")
            
            print()
            print("Predicados por fold:")
            for fname, pdict in folds_dict.items():
                print(f"  {fname}: {len(pdict)} predicados")
        
        return folds_dict
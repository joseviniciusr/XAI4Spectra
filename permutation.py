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
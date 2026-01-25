# generating synthetic spectra with different noise levels
import pandas as pd
import numpy as np
import kennard_stone as ks

# Add parent directory to sys.path so local module 'synthetic' (one level up) can be imported
import sys
from pathlib import Path # for path manipulations
parent_dir = Path.cwd().parent.parent.resolve() # move two levels up from current working directory
if str(parent_dir) not in sys.path: # check to avoid duplicates
    sys.path.insert(0, str(parent_dir)) # insert at the start of sys.path to prioritize local modules

from synthetic import generate_synthetic_spectral_data

config = [
    {
        'nome': 'A',
        'n_amostras': 156,
        'picos': [250, 380, 550, 700, 850],  # 4 picos
        'amp_media': 1.0,
        'amp_std': 0.3,
        'larg_media': 15.0,
        'larg_std': 2.0,
        'ruido_std': 0.04
    },
    {
        'nome': 'B',
        'n_amostras': 146,
        'picos': [50, 250, 380, 550, 850],  # 3 picos (sem pico em 550)
        'amp_media': 1.4,
        'amp_std': 0.5,
        'larg_media': 15.0,
        'larg_std': 1.8,
        'ruido_std': 0.035
    }
]

data_complete = generate_synthetic_spectral_data(
    configuracao_classes=config,
    n_pontos=500,
    x_min=1,
    x_max=1000,
    seed=0
)

# Split dataset by class and create calibration/prediction sets using Kennard-Stone (as in original pipeline)
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.iloc[:, 1:], test_size=0.30)  # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.iloc[:, 1:], test_size=0.30)  # class B
XB_cal = XB_cal.reset_index(drop=True)
XB_pred = XB_pred.reset_index(drop=True)

Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True)  # concatenating both classes
Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
ycalclass = pd.Series(['A']*XA_cal.shape[0] + ['B']*XB_cal.shape[0])  # target for calibration set
ypredclass = pd.Series(['A']*XA_pred.shape[0] + ['B']*XB_pred.shape[0])  # target for prediction set

# Xcalclass_prep = Xcalclass.copy()
# Xpredclass_prep = Xpredclass.copy()

# preprocessings
import preprocessings as prepr  # preprocessing methods
Xcalclass_prep, mean_calclass  = prepr.mc(Xcalclass)
Xpredclass_prep = Xpredclass - mean_calclass

# PLS-DA with optimized latent variables
from modeling import pls_optimized

plsda_results = pls_optimized(
    Xcalclass_prep, 
    ycalclass,
    LVmax=1,
    Xpred=Xpredclass_prep,
    ypred=ypredclass,
    aim='classification',
    cv=10
)

# Convenience references used later
pls_model = plsda_results[3]               # fitted PLS model
vip_scores_mat = plsda_results[4]          # VIP scores matrix (features × LV)
y_pred_cont = plsda_results[5].iloc[:, -1] # continuous predictions for Xcalclass (used for MI/Cov)
spectral_cuts = [
('F1', 1.0, 100.0),
('background1', 100.0, 200.0),
('F2', 200.0, 300.0),
('background2', 300.0, 330.0),
('F3', 330.0, 430.0),
('background3', 430.0, 500.0),
('F4', 500.0, 600.0),
('background4', 600.0, 660.0),
('F5', 660.0, 750.0),
('background5', 750.0, 815.0),
('F6', 815.0, 890.0),
('background6', 890.0, 1000.0)
]

import explaining as exp
spectral_zones_class = exp.extract_spectral_zones(Xcalclass_prep, spectral_cuts)
zone_sums_df = exp.aggregate_spectral_zones(spectral_zones_class, aggregator='extreme')
predicates_quantiles = exp.predicates_by_quantiles(zone_sums_df, [0.2, 0.4, 0.6, 0.8])
co_occurrence_matrix_df = predicates_quantiles[2]
predicate_info_dict = exp.create_predicate_info_dict(
    predicates_df=predicates_quantiles[0],
    predicate_indicator_df=predicates_quantiles[1],
    zone_aggregated_df=zone_sums_df,
    y_predicted_numeric=y_pred_cont
)

# LISTA DE SEMENTES A TESTAR
random_seeds = [0, 1, 2, 3]

all_results_perm = {}
training_samples = len(Xcalclass)

# LOOP: PROCESSAR CADA SEMENTE
y_predicted_numeric = plsda_results[5].iloc[:, -1] # predições numéricas do modelo

for seed in random_seeds:
    print(f"\n{'='*70}")
    print(f"Processando semente: {seed}")
    print(f"{'='*70}\n")
    # Bagging
    bags_result_seed = exp.bagging_predicates(
        zone_sums_df=zone_sums_df,
        y_predicted_numeric=y_predicted_numeric,
        predicates_df=predicates_quantiles[0],
        n_bags=10,
        #n_predicates_per_bag=40,
        n_samples_per_bag=int(training_samples*0.8), # 80 % da base para amostrar (convertido para int)
        min_samples_per_predicate=int(training_samples*0.2), # 20 % da base para limitar (convertido para int)
        replace=False,
        sample_bagging=True,
        predicate_bagging=False,
        random_seed=seed
    )
    # Inserir classe prevista
    for bag_name, pred_dict in bags_result_seed.items(): # iterando sobre cada bag
        for pred_rule, df_info in pred_dict.items():
            df_info['Class_Predicted'] = np.where(df_info['Predicted_Y'] >= 0.5, 'A', 'B') # binarizando com threshold 0.5, A = eut, B = dist
    # Calcular MI

        perm_results = exp.calculate_predicate_metrics_permutation(
        estimator=pls_model,
        Xcalclass_prep=Xcalclass_prep,
        y_calclass=y_pred_cont,
        folds_struct=bags_result_seed,
        predicates_df=predicates_quantiles[0],
        spectral_cuts=spectral_cuts,
        scoring='neg_root_mean_squared_error',
        task_type='regression',
        n_repeats=5,  # Usar 10-20 em produção para resultados mais estáveis
        random_state=seed,
        n_jobs=22,
        verbose=True,
        save_detailed_results=True
    )

    # Remove todos os valores iguais a zero de todos os bags em perm_results[bag]["Permutation"] e salva como perm_results_thresholded
    perm_results_thresholded = {}
    for bag, df in perm_results.items():
        # Verifica se é um DataFrame e se a coluna 'Permutation' existe
        if isinstance(df, pd.DataFrame) and 'Permutation' in df.columns:
            filtered_df = df[df['Permutation'] > 0].copy()
            perm_results_thresholded[bag] = filtered_df
        else:
            # Se não for DataFrame esperado, apenas copia
            perm_results_thresholded[bag] = df


    # Salvar no dicionário principal
    all_results_perm[seed] = {
        'bags_result': bags_result_seed,
        'perm_results_dict': perm_results_thresholded
    }

# CONSTRUÇÃO DE GRAFOS PARA MÚLTIPLAS SEMENTES (LOOP EXTERNO)
graphs_perm_by_seed = {}

for seed in random_seeds:
    print(f"\n{'='*70}")
    print(f"Processando Grafo - Semente: {seed}")
    print(f"{'='*70}\n")
    # Construir grafo para esta semente
    DG = exp.build_predicate_graph(
        bags_result=all_results_perm[seed]['bags_result'],
        mi_results_dict=all_results_perm[seed]['perm_results_dict'],
        co_occurrence_matrix_df=co_occurrence_matrix_df,
        predicates_df=predicates_quantiles[0],
        random_state=seed,
        show_details=True
    )
    # Armazenar grafo
    graphs_perm_by_seed[seed] = DG  

# Calcular LRC usando a função pronta do explaining.py
lrc_perm_by_seed = {}
for seed in random_seeds:
    DG = graphs_perm_by_seed[seed]
    lrc_perm_df_seed = exp.calculate_lrc_single_graph(DG, predicates_quantiles[0])
    lrc_perm_df_seed['Seed'] = seed  # Adicionar coluna com a semente
    lrc_perm_by_seed[seed] = lrc_perm_df_seed

# junando todas as colunas 'Node' de lrc_by_seed em um único dataframe
lrc_perm_all_seeds_df = pd.DataFrame()
for seed in random_seeds:
    lrc_perm_df_seed = lrc_perm_by_seed[seed].rename(columns={'Node': f'Predicate_perm_Seed_{seed}'})
    lrc_perm_all_seeds_df = pd.concat([lrc_perm_all_seeds_df, lrc_perm_df_seed[[f'Predicate_perm_Seed_{seed}']]], axis=1)

# vamos filtrar lrc_by_seed em cada semente para manter apenas as zonas espectrais únicas com maior LRC em um mesmo dataframe
lrc_perm_unique_by_seed = {}
for seed, lrc_df in lrc_perm_by_seed.items():
    lrc_perm_unique_df = lrc_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
    lrc_perm_unique_df = lrc_perm_unique_df.sort_values(by='Local_Reaching_Centrality', ascending=False).reset_index(drop=True)
    lrc_perm_unique_by_seed[seed] = lrc_perm_unique_df

from collections import defaultdict

# Coletar posições de cada predicado em cada seed do lrc_perm_all_seeds_df
positions_dict_lrc = defaultdict(list) # o defaultdict cria listas vazias automaticamente

# Iterar sobre cada coluna do dataframe lrc_perm_all_seeds_df
for col in lrc_perm_all_seeds_df.columns:
    # Pegar os predicados da coluna (não-nulos)
    predicates_in_seed = lrc_perm_all_seeds_df[col].dropna().tolist()
    
    # Para cada predicado, guardar sua posição (1-based)
    for position, predicate in enumerate(predicates_in_seed, start=1):
        positions_dict_lrc[predicate].append(position)

# Calcular média e número de aparições
results_lrc = []
for predicate, positions in positions_dict_lrc.items():
    zone_row = predicates_quantiles[0].loc[predicates_quantiles[0]['rule'] == predicate, 'zone']
    zone_value = zone_row.values[0] if not zone_row.empty else None
    results_lrc.append({
        'Predicate': predicate,
        'Mean_Position': np.mean(positions),
        'Appearances': len(positions),
        'Zone': zone_value
    })

# Ordenar: menor posição média primeiro, mais aparições em caso de empate
ranking_lrc_perm_df = pd.DataFrame(results_lrc).sort_values(
    by=['Mean_Position', 'Appearances'], 
    ascending=[True, False]
).reset_index(drop=True)

# Lista final ordenada
lista_ordenada_lrc = ranking_lrc_perm_df['Predicate'].tolist()
ranking_lrc_perm_df
ranking_perm_lrc_unique_df = ranking_lrc_perm_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)

with pd.ExcelWriter('Permutation_method.xlsx') as writer:
    # Primeiro: salvar os rankings médios
    ranking_lrc_perm_df.to_excel(writer, sheet_name='Mean_LRC', index=False)
    ranking_perm_lrc_unique_df.to_excel(writer, sheet_name='Mean_LRC_Unique', index=False)
    
    # Segundo: salvar os LRCs por seed
    for seed in random_seeds:
        lrc_perm_by_seed[seed].to_excel(writer, sheet_name=f'LRC_seed_{seed}', index=False)
    
    # Terceiro: salvar os resultados de permurbação por bag e seed
    for seed in random_seeds:
        for bag_name, df in all_results_perm[seed]['perm_results_dict'].items():
            # Criar nome único: seed_0_Bag_1, seed_1_Bag_1, etc.
            sheet_name = f'seed_{seed}_{bag_name}'[:31]  # Excel limita a 31 caracteres
            df.to_excel(writer, sheet_name=sheet_name, index=False)
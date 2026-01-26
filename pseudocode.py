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

import pandas as pd
pd.options.plotting.backend = 'plotly' # setting plotly as the backend for pandas plotting 
data_complete.iloc[:, 1:].T.plot()
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
import explaining as exp

seed=42
training_samples = len(Xcalclass)

# LOOP: PROCESSAR CADA SEMENTE
y_predicted_numeric = plsda_results[5].iloc[:, -1] # predições numéricas do modelo

# Bagging
bags_result = exp.bagging_predicates(
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
for bag_name, pred_dict in bags_result.items(): # iterando sobre cada bag
    for pred_rule, df_info in pred_dict.items():
        df_info['Class_Predicted'] = np.where(df_info['Predicted_Y'] >= 0.5, 'A', 'B') # binarizando com threshold 0.5, A = eut, B = dist

    # Calcular MI
cov_results_dict = exp.calculate_predicate_metrics(
    bags_result=bags_result,
    metric='covariance', # covariance ou mutual_information
    threshold=0.01, # threshold para cortar predicados irrelevantes
    n_neighbors=5
)
# Salvar no dicionário principal
all_results_cov = {
    'bags_result': bags_result,
    'cov_results_dict': cov_results_dict
}
DG = exp.build_predicate_graphv2(
    bags_result=all_results_cov['bags_result'],
    predicate_ranking_dict=all_results_cov['cov_results_dict'],
    metric_column='Covariance',  # ou 'Covariance' se mudar a métrica
    random_state=seed,
    show_details=True
)
# Calcular LRC usando a função pronta do explaining.py
lrc_cov_df = exp.calculate_lrc_single_graph(DG, predicates_quantiles[0])


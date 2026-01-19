# Importing the necessary libraries
import pandas as pd
import numpy as np
import kennard_stone as ks
pd.options.plotting.backend = 'plotly'  # setting plotly as the backend for pandas plotting

# Add parent directory to sys.path so local module 'synthetic' (one level up) can be imported
import sys
from pathlib import Path # for path manipulations
parent_dir = Path.cwd().parent.resolve() # get the current working directory (Path.cwd()) and move one level up (parent), returning absolute path (resolve())
if str(parent_dir) not in sys.path: # check to avoid duplicates
    sys.path.insert(0, str(parent_dir)) # insert at the start of sys.path to prioritize local modules

# Loading a soil spectral dataset based on X-ray fluorescence (XRF)
data_complete = pd.read_csv('../XRF_databases/bank_notes/plsda/bank_notes.csv', sep=';') # local copy of Toledo 2022 dataset (os ... indica para omitir o caminho longo)
data = data_complete.loc[:, '1':'26.07']

# Split dataset by class and create calibration/prediction sets using Kennard-Stone (as in original pipeline)
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, '1':'26.07'], test_size=0.30)  # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, '1':'26.07'], test_size=0.30)  # class B
XB_cal = XB_cal.reset_index(drop=True)
XB_pred = XB_pred.reset_index(drop=True)

Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True)  # concatenating both classes
Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
ycalclass = pd.Series(['A']*XA_cal.shape[0] + ['B']*XB_cal.shape[0])  # target for calibration set
ypredclass = pd.Series(['A']*XA_pred.shape[0] + ['B']*XB_pred.shape[0])  # target for prediction set

# preprocessings
import preprocessings as prepr  # preprocessing methods for XRF data

Xcalclass_prep, mean_calclass, mean_calclass_poisson  = prepr.poisson(Xcalclass, mc=True)
Xpredclass_prep = ((Xpredclass/np.sqrt(mean_calclass)) - mean_calclass_poisson)

# PLS-DA with optimized latent variables
from modeling import pls_optimized

plsda_results = pls_optimized(
    Xcalclass_prep, 
    ycalclass,
    LVmax=4,
    Xpred=Xpredclass_prep,
    ypred=ypredclass,
    aim='classification',
    cv=10
)

# Convenience references used later
pls_model = plsda_results[3]               # fitted PLS model
vip_scores_mat = plsda_results[4]          # VIP scores matrix (features × LV)
y_pred_cont = plsda_results[5].iloc[:, -1] # continuous predictions for Xcalclass (used for MI/Cov)

# establishing spectral cuts based on expert knowledge of XRF spectra
spectral_cuts = [
('background1', 1.0, 2.74),
('Ar ka + Ag L', 2.76, 3.47),
('Ca ka', 3.5, 3.91),
('Ca kb', 3.93, 4.24),
('Ti ka', 4.26, 4.72),
('Ti kb', 4.75, 5.13),
('background2', 5.16, 6.12),
('Fe ka', 6.15, 6.76),
('Fe kb', 6.79, 7.32),
('background3', 7.35, 7.78),
('Cu', 7.81, 8.29),
('background4', 8.32, 21.46),
('Ag ka scattering', 21.49, 22.71),
('background5', 22.74, 24.52),
('background6', 24.55, 26.07),
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

import shap

# Para PLSRegression, usamos KernelExplainer porque não há explainer dedicado muito rápido
explainer_pls = shap.KernelExplainer(plsda_results[3].predict, Xcalclass_prep)
shap_values_pls = explainer_pls(Xcalclass_prep)

shap_global_importance = pd.DataFrame({
    'energy': Xpredclass_prep.columns,
    'Mean_Abs_SHAP': np.abs(shap_values_pls.values).mean(axis=0)}) # tomando a importancia global como a media dos valores absolutos dos valores SHAP para cada feature
shap_global_importance.sort_values(by='Mean_Abs_SHAP', ascending=False, inplace=True)

# vamos gerar uma nova coluna em shap_global_importance com o nome da zona espectral correspondente de acordo com a lista spectral_cuts
energy_to_zone_shap = {}
for zone_name, start, end in spectral_cuts:
    for i in shap_global_importance['energy']:
        i_float = float(i)
        if start <= i_float <= end:
            energy_to_zone_shap[i] = zone_name
shap_global_importance['Zone'] = shap_global_importance['energy'].map(energy_to_zone_shap)

# agora vamos filtrar shap_global_importance para manter apenas as zonas espectrais únicas com maior SHAP score
shap_unique_df = shap_global_importance.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
shap_unique_df = shap_unique_df.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
shap_unique_df.to_csv('shap_bank_notes.csv', index=False, sep=';')
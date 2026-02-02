# Importing the necessary libraries
import pandas as pd
import numpy as np
import kennard_stone as ks
pd.options.plotting.backend = 'plotly'  # setting plotly as the backend for pandas plotting

# Add parent directory to sys.path so local module 'synthetic' (one level up) can be imported
import sys
from pathlib import Path # for path manipulations
parent_dir = Path.cwd().parent.parent.resolve() # move two levels up from current working directory
if str(parent_dir) not in sys.path: # check to avoid duplicates
    sys.path.insert(0, str(parent_dir)) # insert at the start of sys.path to prioritize local modules

# Loading a soil spectral dataset based on X-ray fluorescence (XRF)
data_complete = pd.read_csv(f'{parent_dir}/XRF_databases/soil/plsda/soil.csv', sep=';') # local copy of Toledo 2022 dataset (os ... indica para omitir o caminho longo)
data = data_complete.loc[:, '1':'15']
# Split dataset by class and create calibration/prediction sets using Kennard-Stone (as in original pipeline)
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, '1':'15'], test_size=0.30)  # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, '1':'15'], test_size=0.30)  # class B
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

# establishing spectral cuts based on expert knowledge of XRF spectra
spectral_cuts = [
('background1', 1.0, 1.33),
('Al', 1.33, 1.63),
('Si', 1.63, 1.86),
('P', 1.86, 2.10),
('background2', 2.10, 2.19),
('S', 2.19, 2.44),
('background3', 2.44, 2.55),
('Rh L + Ar', 2.55, 3.10),
('background4', 3.10, 3.21),
('K', 3.21, 3.42),
('background5', 3.42, 3.53),
('Ca ka', 3.53, 3.84),
('Ca kb', 3.84, 4.14),
('background6', 4.14, 4.37),
('Ti ka', 4.37, 4.66),
('background7', 4.66, 4.75),
('Ti kb', 4.75, 5.12),
('Cr', 5.12, 5.77),
('Mn', 5.77, 6.02),
('background8', 6.02, 6.13),
('Fe ka', 6.13, 6.68),
('background9', 6.68, 6.80),
('Fe kb', 6.80, 7.30),
('background10', 7.30, 7.91),
('Cu', 7.91, 8.20),
('background11', 8.20, 10.69),
('Fe ka + Ti ka', 10.69, 11.14),
('background12', 11.14, 12.55),
('sum Fe' , 12.55, 13.1),
('background13', 13.1, 15.0)
]

import shap

# Para PLSRegression, usamos KernelExplainer porque não há explainer dedicado muito rápido
explainer_pls = shap.KernelExplainer(plsda_results[3].predict, Xcalclass_prep, njobs=22)
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
shap_unique_df.to_csv('shap_soil.csv', index=False, sep=';')
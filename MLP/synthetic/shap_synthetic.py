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

from modeling import mlp_optimized

mlp_model = mlp_optimized(Xcalclass_prep, ycalclass, Xpredclass_prep, ypredclass, 
                          aim='classification', 
                          hidden_layer_sizes=(64,32),
                          activation='tanh',
                          learning_rate='adaptive',
                          max_iter=10,
                          random_state=1)
# establishing spectral cuts based on expert knowledge of XRF spectra
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

import shap

model_predict_proba = lambda x: mlp_model[3].predict_proba(x)[:, 1] # o 1 é a probabilidade da classe positiva
explainer = shap.KernelExplainer(model_predict_proba, Xcalclass_prep)  # using a subset of calibration data as background for SHAP
shap_exp = explainer(Xcalclass_prep)  # explain a subset of calibration data

shap_values = shap_exp.values
shap_global_importance = pd.DataFrame({
    'energy': Xcalclass_prep.columns,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)})

# vamos gerar uma nova coluna em shap_global_importance com o nome da zona espectral correspondente de acordo com a lista spectral_cuts
# energy_to_zone_shap = {}
# for zone_name, start, end in spectral_cuts:
#     for i in shap_global_importance['energy']:
#         i_float = float(i)
#         if start <= i_float <= end:
#             energy_to_zone_shap[i] = zone_name
# shap_global_importance['Zone'] = shap_global_importance['energy'].map(energy_to_zone_shap)

# # agora vamos filtrar shap_global_importance para manter apenas as zonas espectrais únicas com maior SHAP score
# shap_unique_df = shap_global_importance.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
# shap_unique_df = shap_unique_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
shap_global_importance.to_csv('shap_synthetic.csv', index=False, sep=';')
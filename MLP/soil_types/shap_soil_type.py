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
data_complete = pd.read_csv(f'{parent_dir}/XRF_databases/soil_types/plsda/soil_types.csv', sep=';') # local copy of Toledo and Guarapuava soil datasets
data = data_complete.loc[:, '1':'15']

# Split dataset by class and create calibration/prediction sets using Kennard-Stone (as in original pipeline)
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, '1':'15'], test_size=0.30) # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, '1':'15'], test_size=0.30) # class B
XB_cal = XB_cal.reset_index(drop=True)
XB_pred = XB_pred.reset_index(drop=True)

Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True) # concatenating both classes
Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
ycalclass = pd.Series(['A']*XA_cal.shape[0] + ['B']*XB_cal.shape[0]) # creating the target variable for calibration set
ypredclass = pd.Series(['A']*XA_pred.shape[0] + ['B']*XB_pred.shape[0]) # creating the target variable for prediction set

# preprocessings
import preprocessings as prepr # preprocessing methods for XRF data

Xcalclass_prep, mean_calclass, mean_calclass_poisson  = prepr.poisson(Xcalclass, mc=True)
Xpredclass_prep = ((Xpredclass/np.sqrt(mean_calclass)) - mean_calclass_poisson)


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
('background1', 1.0, 1.33),
('Al', 1.33, 1.63),
('Si', 1.63, 1.86),
('P', 1.86, 2.19),
('S', 2.19, 2.44),
('background2', 2.44, 2.55),
('Rh L + Ar', 2.55, 3.21),
('K', 3.21, 3.53),
('Ca ka', 3.53, 3.84),
('Ca kb', 3.84, 4.14),
('background3', 4.14, 4.37),
('Ti ka', 4.37, 4.66),
('background4', 4.66, 4.75),
('Ti kb', 4.75, 5.12),
('Cr', 5.12, 5.77),
('Mn', 5.77, 6.13),
('Fe ka', 6.13, 6.80),
('Fe kb', 6.80, 7.30),
('background5', 7.30, 7.91),
('Cu', 7.91, 8.20),
('background6', 8.20, 10.69),
('Fe ka + Ti ka', 10.69, 11.14),
('background7', 11.14, 12.55),
('sum Fe' , 12.55, 13.1),
('background8', 13.1, 15.0)
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
energy_to_zone_shap = {}
for zone_name, start, end in spectral_cuts:
    for i in shap_global_importance['energy']:
        i_float = float(i)
        if start <= i_float <= end:
            energy_to_zone_shap[i] = zone_name
shap_global_importance['Zone'] = shap_global_importance['energy'].map(energy_to_zone_shap)

# agora vamos filtrar shap_global_importance para manter apenas as zonas espectrais únicas com maior SHAP score
shap_unique_df = shap_global_importance.sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
shap_unique_df = shap_unique_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
shap_unique_df.to_csv('shap_soil_type.csv', index=False, sep=';')
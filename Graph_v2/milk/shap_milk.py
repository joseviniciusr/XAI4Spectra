# importing the necessary libraries
import pandas as pd
import numpy as np
import kennard_stone as ks

# loading a soil spectral dataset based on X-ray fluorescence (XRF)
data_complete = pd.read_csv('XRF_databases/milk/plsda/milk.csv', sep=';') # local copy of Toledo 2022 dataset
data = data_complete.loc[:, '2.66':'22.62']

# Creating a new column 'Class' based on the condition of the samples in the 'Type' column being 'Authentic'
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, '2.66':'22.62'], test_size=0.30) # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, '2.66':'22.62'], test_size=0.30) # class B
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

from modeling import pls_optimized

# performing PLS-DA with optimized latent variables
plsda_results = pls_optimized(Xcalclass_prep, 
                              ycalclass,
                              LVmax=4,
                              Xpred=Xpredclass_prep,
                              ypred=ypredclass,
                              aim='classification',
                              cv=10)
plsda_results[0]

# establishing spectral cuts based on expert knowledge of XRF spectra
# establishing spectral cuts based on expert knowledge of XRF spectra
spectral_cuts = [
('Ag La', 2.66, 3.10),
('Ag Lb', 3.10, 3.46),
('Ca', 3.46, 3.92),
('background', 3.92, 6.12),
('Fe', 6.12, 6.68),
('Cu', 6.70, 8.37),
('Zn', 8.37, 9.10),
('Bremsstrahlung', 9.10, 20.06),
('Ag compton', 20.06, 21.62),
('Ag ka', 21.48, 22.62)
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
shap_unique_df.to_csv('shap_milk.csv', index=False, sep=';')
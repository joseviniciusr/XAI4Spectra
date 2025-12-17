# Genetic Algorithm (GA) optimization for internal parameters of SMeX incorporating PLS-DA models with XRF spectral libraries

import numpy as np
import pandas as pd
from modeling import pls_optimized
import explaining as exp
import preprocessings as prepr
import kennard_stone as ks

# Instructions dictionary containing dataset-specific parameters
instructions = {
    'datasets': {
        'soil': {
            'spectral_range': ['1', '15'],
            'LV': 4,
            'spectral_cuts': [
                ('background1', 1.0, 1.33),
                ('Al', 1.34, 1.63),
                ('Si', 1.64, 1.86),
                ('P', 1.87, 2.10),
                ('background2', 2.11, 2.19),
                ('S', 2.20, 2.44),
                ('background3', 2.45, 2.55),
                ('Rh L + Ar', 2.56, 3.10),
                ('background4', 3.11, 3.21),
                ('K', 3.22, 3.42),
                ('background5', 3.43, 3.53),
                ('Ca ka', 3.54, 3.84),
                ('Ca kb', 3.92, 4.14),
                ('background6', 4.15, 4.37),
                ('Ti ka', 4.38, 4.66),
                ('background7', 4.67, 4.75),
                ('Ti kb', 4.76, 5.12),
                ('Cr', 5.13, 5.77),
                ('Mn', 5.78, 6.02),
                ('background8', 6.03, 6.13),
                ('Fe ka', 6.14, 6.68),
                ('background9', 6.69, 6.80),
                ('Fe kb', 6.81, 7.30),
                ('background10', 7.31, 7.91),
                ('Cu', 7.92, 8.20),
                ('background11', 8.21, 10.69),
                ('Fe ka + Ti ka', 10.7, 11.14),
                ('background12', 11.15, 12.55),
                ('sum Fe', 12.56, 13.1),
                ('background13', 13.11, 15.0)
            ]
        },
        'bank_notes': {
            'spectral_range': ['1', '26.07'],
            'LV': 4,
            'spectral_cuts': [
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
                ('Ti ka', 24.55, 26.07)
            ]
        }
    }
}

################################## DATA LOADING, PREPROCESSING, AND MODELING ################################################################################################################

# selecting the dataset to be used
dataset_target = 'soil'  # selecting the dataset to be used

# loading a soil spectral dataset based on X-ray fluorescence (XRF)
data_complete = pd.read_csv(f'XRF_databases/{dataset_target}/plsda/{dataset_target}.csv', sep=';') 
data = data_complete.loc[:, instructions['datasets'][dataset_target]['spectral_range'][0]:instructions['datasets'][dataset_target]['spectral_range'][1]]

# Creating a new column 'Class' based on the condition of 'BSP' values
data_A = data_complete[data_complete['Class'] == 'A'].reset_index(drop=True)
data_B = data_complete[data_complete['Class'] == 'B'].reset_index(drop=True)

# splitting the data into calibration and prediction sets by kennard-stone algorithm
XA_cal, XA_pred = ks.train_test_split(data_A.loc[:, instructions['datasets'][dataset_target]['spectral_range'][0]:instructions['datasets'][dataset_target]['spectral_range'][1]], test_size=0.30) # class A
XA_cal = XA_cal.reset_index(drop=True)
XA_pred = XA_pred.reset_index(drop=True)

XB_cal, XB_pred = ks.train_test_split(data_B.loc[:, instructions['datasets'][dataset_target]['spectral_range'][0]:instructions['datasets'][dataset_target]['spectral_range'][1]], test_size=0.30) # class B
XB_cal = XB_cal.reset_index(drop=True)
XB_pred = XB_pred.reset_index(drop=True)

Xcalclass = pd.concat([XA_cal, XB_cal], axis=0).reset_index(drop=True) # concatenating both classes
Xpredclass = pd.concat([XA_pred, XB_pred], axis=0).reset_index(drop=True)
ycalclass = pd.Series(['A']*XA_cal.shape[0] + ['B']*XB_cal.shape[0]) # creating the target variable for calibration set
ypredclass = pd.Series(['A']*XA_pred.shape[0] + ['B']*XB_pred.shape[0]) # creating the target variable for prediction set

import preprocessings as prepr # preprocessing methods for XRF data

Xcalclass_prep, mean_calclass, mean_calclass_poisson  = prepr.poisson(Xcalclass, mc=True) # applying poisson pretreatment with mean centering
Xpredclass_prep = ((Xpredclass/np.sqrt(mean_calclass)) - mean_calclass_poisson) # applying the same preprocessing to prediction set

from modeling import pls_optimized

# performing PLS-DA with optimized latent variables
plsda_results = pls_optimized(Xcalclass_prep, 
                              ycalclass,
                              LVmax=instructions['datasets'][dataset_target]['LV'],
                              Xpred=Xpredclass_prep,
                              ypred=ypredclass,
                              aim='classification',
                              cv=10)
model_info = plsda_results[0] # saving model information

# spectral cuts for VIP, SHAP, and SMeX implementation
spectral_cuts = instructions['datasets'][dataset_target]['spectral_cuts']

############################### EXPLAINABILITY ANALYSES #################################################################################################################################################################################

# VIP scores
vip_scores_df = pd.DataFrame({
    'energy' : plsda_results[4].T.index,
    'VIP_Score' : plsda_results[4].T.iloc[:,0].values
})
vip_scores_df = vip_scores_df.sort_values(by='VIP_Score', ascending=False).reset_index(drop=True)

# generating a new column in vip_scores_df with the name of the corresponding spectral zone according to the spectral_cuts list
energy_to_zone_vip = {} # dictionary to map energy to spectral zone
for zone_name, start, end in spectral_cuts: # iterating over each spectral zone (which has name, start, and end)
	for i in vip_scores_df['energy']:
		i_float = float(i)
		if start <= i_float <= end:
			energy_to_zone_vip[i] = zone_name
vip_scores_df['Zone'] = vip_scores_df['energy'].map(energy_to_zone_vip) # mapping the 'energy' values to their corresponding zones using the energy_to_zone_vip dictionary

# Filtraring vip_scores_df to keep only unique spectral zones with the highest VIP score
vip_scores_unique_df = vip_scores_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
vip_scores_unique_df = vip_scores_unique_df.sort_values(by='VIP_Score', ascending=False).reset_index(drop=True)

# SMeX GA OPTIMIZATION

from deap import creator, base, tools, algorithms
import random

rseed = 42
random.seed(rseed) # setting random seed for reproducibility

pop_size = 200 # population size
num_generations = 100 # number of generations
crossover_prob = 0.7 # crossover probability
mutation_prob = 0.1 # mutation probability

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # fitness function to be maximized
creator.create("Individual", list, fitness=creator.FitnessMax) # individual representation

# registring functions to create individuals and population
toolbox = base.Toolbox()

# parameters to be optimized

# agregate function to be used in SMeX
toolbox.register("attr_agregate_function", lambda: random.choice(['sum', 'median', 'max']))

# number of bags
toolbox.register("attr_nbags", random.randint, 20, 150) # number of bags between 20 and 150

# number of samples per bag as a fraction of the total samples
toolbox.register("attr_n_samples_per_bag_frac", random.uniform, 0.5, 0.9) # fraction between 0.5 and 0.9

# minimum number of samples per predicate as a fraction of the total samples
toolbox.register("attr_min_samples_per_predicate_frac", random.uniform, 0.2, 0.6) # fraction between 0.2 and 0.4

# if replacement is used when sampling
toolbox.register("attr_replacement", lambda: random.choice([True, False]))

# if the bagging will be applied to the predicates
#toolbox.register("attr_bagging_on_predicates", lambda: random.choice([True, False]))

# creating an individual by combining all attributes
toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_agregate_function,
                    toolbox.attr_nbags,
                    toolbox.attr_n_samples_per_bag_frac,
                    toolbox.attr_min_samples_per_predicate_frac,
                    toolbox.attr_replacement),
                    #toolbox.attr_bagging_on_predicates),
                    n=1)

# creating the population
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# crossover genetic operator
toolbox.register("mate", tools.cxUniform, indpb=0.5) # uniform crossover

# mutation operator
def mutate_individual(individual):
    # Mutate agregate_function
    if random.random() < 0.25:
        individual[0] = random.choice(['sum', 'median', 'max'])
    
    # Mutate nbags
    if random.random() < 0.25:
        individual[1] = random.randint(20, 150)
    
    # Mutate n_samples_per_bag
    if random.random() < 0.25:
        individual[2] = random.uniform(0.5, 0.9)
    
    # Mutate min_samples_per_predicate
    if random.random() < 0.25:
        individual[3] = random.uniform(0.05, 0.4)
    
    # Mutate replacement
    if random.random() < 0.25:
        individual[4] = random.choice([True, False])
    
    # Mutate bagging_on_predicates
    # if random.random() < 0.25:
    #     individual[5] = random.choice([True, False])
    
    return individual,

toolbox.register("mutate", mutate_individual)

# selection operator
toolbox.register("select", tools.selTournament, tournsize=5)

# fitness evaluation function - CORRIGIDA com tratamento de erro
def RBO_evaluate(individual):
    try:
        # extracting individual parameters
        agregate_function = individual[0] # 'sum', 'median', or 'max'
        nbags = individual[1] # integer number of bags
        n_samples_per_bag_frac = individual[2] # fraction of samples per bag
        min_samples_per_predicate_frac = individual[3] # fraction of minimum samples per predicate
        replacement = individual[4] # boolean for replacement
        #bagging_on_predicates = individual[5] # boolean for bagging on predicates

        spectral_zones_class = exp.extract_spectral_zones(Xcalclass, spectral_cuts)
        zone_sums_df = exp.aggregate_spectral_zones(spectral_zones_class, aggregator=agregate_function)
        predicates_quantiles = exp.predicates_by_quantiles(zone_sums_df, [0.2, 0.4, 0.6, 0.8])
        co_occurrence_matrix_df = predicates_quantiles[2]

        training_samples = len(Xcalclass)
        y_predicted_numeric = plsda_results[5].iloc[:, -1]

        seed = rseed
            
        # Bagging
        bags_result = exp.bagging_predicates(
            zone_sums_df=zone_sums_df,
            y_predicted_numeric=y_predicted_numeric,
            predicates_df=predicates_quantiles[0],
            n_bags=nbags,
            n_samples_per_bag=int(training_samples * n_samples_per_bag_frac),
            min_samples_per_predicate=int(training_samples * min_samples_per_predicate_frac),
            replace=replacement,
            sample_bagging=True,
            predicate_bagging=False, #bagging_on_predicates,
            random_seed=seed
        )
        
        # Inserir classe prevista
        for bag_name, pred_dict in bags_result.items():
            for pred_rule, df_info in pred_dict.items():
                df_info['Class_Predicted'] = np.where(df_info['Predicted_Y'] >= 0.5, 'A', 'B')
        
        # Calcular MI
        mi_results_dict_seed = exp.calculate_predicate_metrics(
            bags_result=bags_result,
            metric='mutual_info',
            threshold=0.1,
            n_neighbors=5
        )
        
        # Construir grafo com show_details=False para evitar output excessivo
        DG = exp.build_predicate_graph(
            bags_result=bags_result,
            mi_results_dict=mi_results_dict_seed,
            co_occurrence_matrix_df=co_occurrence_matrix_df,
            predicates_df=predicates_quantiles[0],
            random_state=seed,
            show_details=False
        )

        # Calcular LRC
        import networkx as nx
        
        local_reaching_centrality = {
            node: nx.local_reaching_centrality(DG, node, weight='weight') 
            for node in DG.nodes()
        }

        # Ordenar por LRC
        sorted_lrc = sorted(local_reaching_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Criar DataFrame com LRC
        lrc_df = pd.DataFrame(sorted_lrc, columns=['Node', 'Local_Reaching_Centrality'])
        
        # Extrair informações dos predicados
        zones = []
        
        for node in lrc_df['Node']:
            if node.startswith('Class_'):
                zones.append(None)
            else:
                pred_row = predicates_quantiles[0][predicates_quantiles[0]['rule'] == node]
                if len(pred_row) > 0:
                    zones.append(pred_row.iloc[0]['zone'])
                else:
                    zones.append(None)
        
        lrc_df['Zone'] = zones
        
        # Filtrar zonas únicas
        lrc_unique_df = lrc_df.drop_duplicates(subset=['Zone'], keep='first').reset_index(drop=True)
        lrc_unique_df = lrc_unique_df[lrc_unique_df['Zone'].notna()]  # Remover None
        lrc_unique_df = lrc_unique_df.sort_values(by='Local_Reaching_Centrality', ascending=False).reset_index(drop=True)
        
        import rbo
        
        # Calcular RBO
        vip_list = vip_scores_unique_df['Zone'].tolist()
        lrc_list = lrc_unique_df['Zone'].tolist()
        rbo_score = rbo.RankingSimilarity(vip_list, lrc_list).rbo(p=0.7, k=10)
        
        return (rbo_score,)
    
    except Exception as err:
        # Em caso de erro, retornar fitness 0 e imprimir o erro
        print(f"ERRO na avaliação: {str(err)}")
        print(f"Parâmetros: agg={individual[0]}, nbags={individual[1]}, "
              f"sample_frac={individual[2]:.2f}, predicate_frac={individual[3]:.2f}, "
              f"replace={individual[4]}")
              #f"replace={individual[4]}, bag_preds={individual[5]}")
        return (0.0,)

# registrando a função de avaliação no toolbox
toolbox.register("evaluate", RBO_evaluate)

# setting up statistics to be recorded and hall of fame
statistics = tools.Statistics(lambda ind: ind.fitness.values)
statistics.register("mean", np.mean)
statistics.register("std", np.std)
statistics.register("var", np.var)
statistics.register("min", np.min)
statistics.register("max", np.max)

hall_of_fame = tools.HallOfFame(20) # keeping the top 5 individuals

# criando a população inicial
population = toolbox.population(n=pop_size)

# excecutando a busca evolutiva via algoritmo genético
print(f"Início do Processo Evolutivo (Pop: {pop_size}, Gens: {num_generations}) ---")
print("=" * 80)

pop, log = algorithms.eaSimple(population,
                               toolbox,
                               cxpb=crossover_prob,
                               mutpb=mutation_prob,
                               ngen=num_generations,
                               stats=statistics,
                               halloffame=hall_of_fame,
                               verbose=True)

print("=" * 80)
print("Fim da Evolução\n")

# interpretando os resultados
print("="*80)
print("MELHORES INDIVÍDUOS ENCONTRADOS:")
print("="*80)
for i, individual in enumerate(hall_of_fame):
    print(f"\n Ranking #{i+1}")
    print(f"   Fitness (RBO Score): {individual.fitness.values[0]:.4f}")
    print(f"   Parâmetros:")
    print(f"      • Agregador: {individual[0]}")
    print(f"      • N° Bags: {individual[1]}")
    print(f"      • Fração amostras/bag: {individual[2]:.2f}")
    print(f"      • Fração min amostras/predicado: {individual[3]:.2f}")
    print(f"      • Replacement: {individual[4]}")
    #print(f"      • Bagging em predicados: {individual[5]}")

# convertendo o hall_of_fame (os melhores indivíduos ao longo de toda a evolução) em um DataFrame e salvando como CSV
hof_df = pd.DataFrame([{
    'Seed' : rseed,
    'Rank': i+1,
    'Fitness_RBO_Score': individual.fitness.values[0],
    'Agregador': individual[0],
    'N_Bags': individual[1],
    'Frac_Samples_per_Bag': individual[2],
    'Frac_Min_Samples_per_Predicate': individual[3],
    'Replacement': individual[4]
} for i, individual in enumerate(hall_of_fame)])
hof_df.to_csv(f'XRF_databases/{dataset_target}/plsda/smeX_ga_optimization_hof.csv', index=False, sep=';')

# salvando as estatísticas do processo evolutivo
statistics_df = pd.DataFrame(log)
statistics_df.to_csv(f'XRF_databases/{dataset_target}/plsda/smeX_ga_optimization_statistics.csv', index=False, sep=';')
import torch 
import numpy as np
import torch.nn as nn
from utils_rf_prove import *
import time
import yaml
import argparse
import warnings
warnings.filterwarnings("ignore")



if __name__ == '__main__':

    # parse config file
    parser = argparse.ArgumentParser(description='rf-prove parameters')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    config_path = parser.parse_args().config_path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) 
    
    
    # Load model and input domain from config
    name_model = config['model']['name']
    model_ori = torch.load(f'../models/{name_model}')
    input_domain = np.array(config['specification']['input_domain'])
    
    if name_model == "cartpole.pt":
        model = BinaryDecisionWrapperCartPole(model_ori)
    elif name_model == "lunarlander.pt":
        model = BinaryDecisionWrapperLunarLander(model_ori)
    elif name_model == "dubinsrejoin.pt":
        model = BinaryDecisionWrapperDubinsrejoin(model_ori)
    else:
        model = model_ori
    
  

    # rf-prove parameters
    seeds = config['rf_prove_params']['seeds']
    training_samples = config['rf_prove_params']['training_samples']
    n_decision_trees = config['rf_prove_params']['n_decision_trees']
    max_depth = config['rf_prove_params']['max_depth']
    heuristic_rf = config['rf_prove_params']['heuristic_rf']
    samples_evaluation = config['rf_prove_params']['sample_size_evaluation'] 
    desired_coverage = config['specification']['desired_coverage']
    verbose = config['rf_prove_params']['verbose']
    wilks_samples = config['rf_prove_params']['wilks_samples']
    digits_to_consider = config['rf_prove_params']['digits_to_consider'] 
    compute_min_side=False 
   

    # Run RF-ProVe
    positive_rates = []
    polytopes = []
    coverages = []
    times = []
    errors = []
    min_sides = []

    for seed in seeds:
        print(f"\n{CYAN_COL}[monitor]{RESET_COL} Running RF-ProVe with seed {seed}...")
        np.random.seed(seed)
        torch.manual_seed(seed)
        param_rf = [training_samples, n_decision_trees, max_depth, heuristic_rf, seed, digits_to_consider]
       
 
        start_time = time.time()

        estimated_positive_rate, polys, coverage, error, min_side = verify(model, input_domain, param_rf, safe_regions=True, save_regions=False, verbose=verbose, sample_size_evaluation=samples_evaluation, desired_coverage=desired_coverage, wilks_filtering=True, wilks_samples=wilks_samples, digits_to_consider=digits_to_consider, compute_min_side=compute_min_side)
        
        positive_rates.append(estimated_positive_rate)
        polytopes.append(polys)
        coverages.append(coverage)
        times.append(time.time() - start_time)
        errors.append(error)
        min_sides.append(min_side)

    
    
    print("=" * 50)
    print(f"\n{BLUE_COL}[monitor]{RESET_COL} Estimate input space positive portion: {np.mean(np.array(positive_rates))}%")
    #print(f"\t{GREEN_COL}[info]{RESET_COL} mean # positive polytopes: {int(np.mean(np.array(polytopes)))}")
    print(f"\t{GREEN_COL}[info]{RESET_COL} Mean # positive polytopes: {np.min(np.array(polytopes))}")
    print(f"\t{GREEN_COL}[info]{RESET_COL} Estimated coverage volume of boxes: {np.mean(np.array(coverages))}%")
    print(f"\t{GREEN_COL}[info]{RESET_COL} Mean error in boxes: {round(np.mean(np.array(errors)),2)}%")
    if compute_min_side: print(f"\t{GREEN_COL}[info]{RESET_COL} Mean min side of boxes: {np.mean(np.array(min_sides))}")
    print(f"\t{RED_COL}[info]{RESET_COL} Mean computaiton Time: {round(np.mean(np.array(times)),2)} seconds - {round(np.mean(np.array(times))/60, 2)} minutes\n")

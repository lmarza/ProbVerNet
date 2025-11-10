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
   
    mean_polytope = []
    mean_coverage = []
    mean_time = []
    mean_error = []

    for model in name_model:
        print(f"\n{CYAN_COL}[monitor]{RESET_COL} Running RF-ProVe with model {model}...")

        model_ori = torch.load(f'models/{model}')
        model = BinaryDecisionWrapperVCAS(model_ori)
        input_domain = np.array(config['specification']['input_domain'])

        # Run RF-ProVe
        positive_rates = []
        polytopes = []
        coverages = []
        times = []
        errors = []


        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            param_rf = [training_samples, n_decision_trees, max_depth, heuristic_rf, seed, digits_to_consider]
        
    
            start_time = time.time()

            estimated_positive_rate, polys, coverage, error, _ = verify(model, input_domain, param_rf, safe_regions=True, save_regions=False, verbose=verbose, sample_size_evaluation=samples_evaluation, desired_coverage=desired_coverage, wilks_filtering=True, wilks_samples=wilks_samples, digits_to_consider=digits_to_consider)
            
            positive_rates.append(estimated_positive_rate)
            polytopes.append(polys)
            coverages.append(coverage)
            times.append(time.time() - start_time)
            errors.append(error)
        

        mean_polytope.append(np.min(np.array(polytopes)))
        mean_coverage.append(np.min(np.array(coverages)))
        mean_time.append(np.min(np.array(times)))
        mean_error.append(np.min(np.array(errors)))

    
    #print(mean_polytope)
    print("=" * 50)
    print(f"\n{BLUE_COL}[monitor]{RESET_COL} Estimate input space positive portion: {np.mean(np.array(positive_rates))}%")
    print(f"\t{GREEN_COL}[info]{RESET_COL} mean # positive polytopes: {int(np.mean(np.array(mean_polytope)))}")
    print(f"\t{GREEN_COL}[info]{RESET_COL} Estimated coverage volume of boxes {np.mean(np.array(mean_coverage))}%")
    print(f"\t{GREEN_COL}[info]{RESET_COL} mean error in the boxes {round(np.mean(np.array(mean_error)),2)}%")
    print(f"\t{RED_COL}[info]{RESET_COL} Mean computaiton Time: {round(np.mean(np.array(mean_time)),2)} seconds - {round(np.mean(np.array(times))/60, 2)} minutes\n")

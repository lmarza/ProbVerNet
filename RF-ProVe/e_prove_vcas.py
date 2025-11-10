import torch
import numpy as np
import time
import yaml
import argparse
import warnings
from utils_rf_prove import BinaryDecisionWrapperVCAS
from utils_e_prove import *
warnings.filterwarnings("ignore")



CYAN_COL = '\033[96m'
BLUE_COL = '\033[94m'
RED_COL = '\033[91m'
GREEN_COL = '\033[92m'
YELLOW_COL = '\033[93m'
RESET_COL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


if __name__ == "__main__":

	# parse config file
	parser = argparse.ArgumentParser(description='rf-prove parameters')
	parser.add_argument('--config_path', type=str, help='Path to the configuration file')
	config_path = parser.parse_args().config_path
	with open(config_path, 'r') as file:
		config = yaml.safe_load(file) 


	# e-prove parameters
	name_model = config['model']['name']
	seeds = config['e_prove_params']['seeds']
	verbose = config['e_prove_params']['verbose']
	wilks_samples = config['e_prove_params']['wilks_samples']
	desired_coverage = config['specification']['desired_coverage']

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
			
		
			start_time = time.time()

			e_prove = eProVe(model, input_domain, wilks_samples, desired_coverage=desired_coverage)
			estimated_positive_rate, info = e_prove.verify()
			
			positive_rates.append(estimated_positive_rate)
			polytopes.append(info['areas-number'])
			coverages.append(info['coverage'])
			times.append(time.time() - start_time)
			errors.append(info['error'])
        

		mean_polytope.append(np.min(np.array(polytopes)))
		mean_coverage.append(np.min(np.array(coverages)))
		mean_time.append(np.min(np.array(times)))
		mean_error.append(np.min(np.array(errors)))
    
	print(mean_polytope)
	print("=" * 50)
	print(f"\n{BLUE_COL}[monitor]{RESET_COL} Estimate input space positive portion: {np.mean(np.array(positive_rates))}%")
	print(f"\t{GREEN_COL}[info]{RESET_COL} mean # positive polytopes: {int(np.mean(np.array(mean_polytope)))}")
	print(f"\t{GREEN_COL}[info]{RESET_COL} Estimated coverage volume of boxes {np.mean(np.array(mean_coverage))}%")
	print(f"\t{GREEN_COL}[info]{RESET_COL} mean % error in the boxes {round(np.mean(np.array(mean_error)),2)}%")
	print(f"\t{RED_COL}[info]{RESET_COL} Mean computaiton Time: {round(np.mean(np.array(mean_time)),2)} seconds - {round(np.mean(np.array(times))/60, 2)} minutes\n")

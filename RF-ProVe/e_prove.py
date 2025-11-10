import torch
import numpy as np
import time
import yaml
import argparse
import warnings
from utils_rf_prove import BinaryDecisionWrapperCartPole, BinaryDecisionWrapperLunarLander, BinaryDecisionWrapperDubinsrejoin
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


	# Load model and input domain from config
	name_model = config['model']['name']
	model_ori = torch.load(f'models/{name_model}')
	input_domain = np.array(config['specification']['input_domain'])

	if name_model == "cartpole.pt":
		model = BinaryDecisionWrapperCartPole(model_ori)
	elif name_model == "lunarlander.pt":
		model = BinaryDecisionWrapperLunarLander(model_ori)
	elif name_model == "dubinsrejoin.pt":
		model = BinaryDecisionWrapperDubinsrejoin(model_ori)

	# rf-prove parameters
	seeds = config['e_prove_params']['seeds']
	verbose = config['e_prove_params']['verbose']
	wilks_samples = config['e_prove_params']['wilks_samples']
	desired_coverage = config['specification']['desired_coverage']



	positive_rates = []
	polytopes = []
	coverages = []
	times = []
	errors = []

	for seed in seeds:
		print(f"\n{CYAN_COL}[monitor]{RESET_COL} Running RF-ProVe with seed {seed}...")
		np.random.seed(seed)
		torch.manual_seed(seed)
		
	
		start_time = time.time()

		e_prove = eProVe(model, input_domain, point_cloud=wilks_samples, desired_coverage=desired_coverage)
		estimated_positive_rate, info = e_prove.verify()
		
		positive_rates.append(estimated_positive_rate)
		polytopes.append(info['areas-number'])
		coverages.append(info['coverage'])
		times.append(time.time() - start_time)
		errors.append(info['error'])


	print("=" * 50)
	print(f"\n{BLUE_COL}[monitor]{RESET_COL} Estimate input space positive portion: {np.mean(np.array(positive_rates))}%")
	print(f"\t{GREEN_COL}[info]{RESET_COL} mean # positive polytopes: {int(np.mean(np.array(polytopes)))}")
	print(f"\t{GREEN_COL}[info]{RESET_COL} Estimated coverage volume of boxes {np.mean(np.array(coverages))}%")
	print(f"\t{GREEN_COL}[info]{RESET_COL} error of boxes {np.mean(np.array(errors))}%")
	print(f"\t{RED_COL}[info]{RESET_COL} Mean computaiton Time: {round(np.mean(np.array(times)),2)} seconds - {round(np.mean(np.array(times))/60, 2)} minutes\n")

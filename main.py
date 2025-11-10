import warnings; warnings.filterwarnings("ignore")
import torch, time, csv
from scripts.eProVe import eProVe

CYAN_COL = '\033[96m'
BLUE_COL = '\033[94m'
RED_COL = '\033[91m'
GREEN_COL = '\033[92m'
YELLOW_COL = '\033[93m'
RESET_COL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

if __name__ == "__main__":

	# Property Configuration (basic model)
	from template.networks import ScalabilityNet as Net
	model_name = "models-basic/model_2_68.pth"
	domain = [[0.0, 1.0], [0.0, 1.0]]

	# Loading the model
	network = torch.load( f"models/{model_name}" ); network.eval()

	# Parameters Configuration
	ratio = .995
	point_cloud = 3500

	print('Starting the computation with ε-ProVe...')

	# Initialize the verification tool and start the verification
	starting_time = time.time()
	prove = eProVe( network, domain, point_cloud)
	lower_bound_safe, info =  prove.verify()

	# Computation of the confidence level
	confidence = (1-(ratio**point_cloud))**info['areas-number']

	# Monitoring information
	starting_time_estimation = time.time()
	real_value = prove.estimate_safe_rate()
	estimation_error = abs(real_value - lower_bound_safe) / real_value

	print()
	print( f"{BLUE_COL}The provably safe portion of the input is of size {BOLD}{lower_bound_safe*100:5.4f}%{RESET_COL}")
	print( f"\t{GREEN_COL}[info]{RESET_COL} Point Cloud Size: {point_cloud}")
	print( f"\t{GREEN_COL}[info]{RESET_COL} Time: {(time.time() - starting_time):3.1f} seconds")
	print( f"\t{GREEN_COL}[info]{RESET_COL} Number Areas: {info['areas-number']}")
	print( f"\t{GREEN_COL}[info]{RESET_COL} Depth Reached: {info['depth-reached']}")
	print( f"\t{RED_COL}[monitor]{RESET_COL} Real safe portion: {real_value*100:5.4f}%")
	print( f"\t{RED_COL}[monitor]{RESET_COL} Error: {estimation_error*100:4.2f}%")
	print( f"\t{RED_COL}[monitor]{RESET_COL} Time: {(time.time() - starting_time_estimation):3.1f} seconds")
	print()
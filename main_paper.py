import warnings; warnings.filterwarnings("ignore")
import torch, time, csv, tqdm
import numpy as np; np.random.seed(2)
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

	csv_file = open( "full_results.csv", 'w')
	csv_writer = csv.writer( csv_file )
	csv_writer.writerow( ["Model", "num-areas (#)", "safe_rate", "Time (s)", "Monitor (%)", "time_monitor", "underestimation (%)"] )

	print( f"\t{GREEN_COL}Starting collectiong all the results with ε-ProVe!{RESET_COL}")

	from template.networks import ScalabilityNet as Net
	for model_code in tqdm.tqdm(['20', '56', '68']):
		model_name = f"models-basic/model_2_{model_code}.pth"
		domain = [[0.0, 1.0], [0.0, 1.0]]

		# Loading the model
		network = torch.load( f"models/{model_name}" ); network.eval()

		# Parameters Configuration
		ratio = .995
		point_cloud = 3500

		# Initialize the verification tool and start the verification
		starting_time = time.time()
		prove = eProVe( network, domain, point_cloud)
		lower_bound_safe, info =  prove.verify()

		# Computation of the confidence level
		confidence = (1-(ratio**point_cloud))**info['areas-number']

		# Monitoring information
		time_monitor = time.time()
		real_value = prove.estimate_safe_rate()
		estimation_error = abs(real_value - lower_bound_safe) / real_value

		# Save to CSV file
		csv_name = f"model_2_{model_code}"
		csv_num_areas = info['areas-number']
		csv_safe_rate = np.round(lower_bound_safe*100, 4)
		csv_time = np.round(time.time() - starting_time, 1)
		
		csv_monitor =  np.round(real_value*100, 4)
		csv_time_monitor = np.round(time.time() - time_monitor, 1)
		csv_underestimation =  np.round(estimation_error*100, 2)
		
		#print([csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation])
		csv_writer.writerow( [csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation] )


	print( f"\t{GREEN_COL}Finished experiments on toy DNNs!\n")
	print( f"\t{GREEN_COL}Starting with Mapless Navigation (MN) DNNs...{RESET_COL}")

	from template.networks import TurtleNetFront as Net
	for model_code in tqdm.tqdm(['1', '2', '3']):
		model_name = f"models-turtle-front/turtle_m{model_code}.pth"
		domain = [[0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0],[0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1.0], [0.95, 1], [0.95, 1.0], [0.95, 1.0],[0.95, 1.0], [0.95, 1.0],[0.95, 1.0], [0.95, 1.0], [-1, 1], [-1, 1]]
		domain[10] = [0, 0.05]

		# Loading the model
		network = torch.load( f"models/{model_name}" ); network.eval()

		# Parameters Configuration
		ratio = .995
		point_cloud = 3500

		# Initialize the verification tool and start the verification
		starting_time = time.time()
		prove = eProVe( network, domain, point_cloud)
		lower_bound_safe, info =  prove.verify()

		# Computation of the confidence level
		confidence = (1-(ratio**point_cloud))**info['areas-number']

		# Monitoring information
		time_monitor = time.time()
		real_value = prove.estimate_safe_rate()
		estimation_error = abs(real_value - lower_bound_safe) / real_value

		# Save to CSV file
		csv_name = f"model_MN_{model_code}"
		csv_num_areas = info['areas-number']
		csv_safe_rate = np.round(lower_bound_safe*100, 4)
		csv_time = np.round(time.time() - starting_time, 1)
		
		csv_monitor =  np.round(real_value*100, 4)
		csv_time_monitor = np.round(time.time() - time_monitor, 1)
		csv_underestimation =  np.round(estimation_error*100, 2)
		
		#print([csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation])
		csv_writer.writerow( [csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation] )


	print( f"\t{GREEN_COL}Finished experiments on MN DNNs!\n")
	print( f"\t{GREEN_COL}Starting with ACAS xu prop 2...{RESET_COL}")

	from template.networks import AcasNetP2 as Net
	for model_code in tqdm.tqdm(['2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '2_9', '3_1', '3_3', '3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '4_1', '4_3', '4_4', '4_5', '4_6', '4_7', '4_8', '4_9', '5_1', '5_2', '5_4', '5_5', '5_6', '5_7', '5_8', '5_9']):
		
		model_name = f"models-acas-p2/ACASXU_run2a_{model_code}_batch_2000.pth"
		domain = [
					[ 0.600000,  0.679858],
					[-0.500000,  0.500000],
					[-0.500000,  0.500000],
					[ 0.450000,  0.500000],
					[-0.500000, -0.450000]
				]
		

		# Loading the model
		network = torch.load( f"models/{model_name}" ); network.eval()

		# Parameters Configuration
		ratio = .995
		point_cloud = 3500

		# Initialize the verification tool and start the verification
		starting_time = time.time()
		prove = eProVe( network, domain, point_cloud)
		lower_bound_safe, info =  prove.verify()

		# Computation of the confidence level
		confidence = (1-(ratio**point_cloud))**info['areas-number']

		# Monitoring information
		time_monitor = time.time()
		real_value = prove.estimate_safe_rate()
		estimation_error = abs(real_value - lower_bound_safe) / real_value

		# Save to CSV file
		csv_name = f"ACAS_{model_code}"
		csv_num_areas = info['areas-number']
		csv_safe_rate = np.round(lower_bound_safe*100, 4)
		csv_time = np.round(time.time() - starting_time, 1)
		
		csv_monitor =  np.round(real_value*100, 4)
		csv_time_monitor = np.round(time.time() - time_monitor, 1)
		csv_underestimation =  np.round(estimation_error*100, 2)
		
		#print([csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation])
		csv_writer.writerow( [csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation] )
		
	print( f"\t{GREEN_COL}Finished experiments on ACAS xu prop 2!\n")
	print( f"\t{GREEN_COL}Starting last experiment on ACAS xu prop 3...{RESET_COL}")

	from template.networks import AcasNetP3 as Net
	for model_code in tqdm.tqdm(['1_3', '1_4', '1_5']):
		
		model_name = f"models-acas-p2/ACASXU_run2a_{model_code}_batch_2000.pth"
		domain = [
			[-0.303531,  -0.298553],
			[-0.009549,   0.009549],
			[ 0.493380,   0.500000],
			[ 0.300000,   0.500000],
			[ 0.300000,  -0.450000]
		]
		

		# Loading the model
		network = torch.load( f"models/{model_name}" ); network.eval()

		# Parameters Configuration
		ratio = .995
		point_cloud = 3500

		# Initialize the verification tool and start the verification
		starting_time = time.time()
		prove = eProVe( network, domain, point_cloud)
		lower_bound_safe, info =  prove.verify()

		# Computation of the confidence level
		confidence = (1-(ratio**point_cloud))**info['areas-number']

		# Monitoring information
		time_monitor = time.time()
		real_value = prove.estimate_safe_rate()
		estimation_error = abs(real_value - lower_bound_safe) / real_value

		# Save to CSV file
		csv_name = f"ACAS_{model_code}"
		csv_num_areas = info['areas-number']
		csv_safe_rate = np.round(lower_bound_safe*100, 4)
		csv_time = np.round(time.time() - starting_time, 1)
		
		csv_monitor =  np.round(real_value*100, 4)
		csv_time_monitor = np.round(time.time() - time_monitor, 1)
		csv_underestimation =  np.round(estimation_error*100, 2)
		
		#print([csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation])
		csv_writer.writerow( [csv_name,  csv_num_areas, csv_safe_rate, csv_time, csv_monitor, csv_time_monitor, csv_underestimation] )


	print( f"\n\n\t{GREEN_COL}======== Finished all the experiments! ========{RESET_COL}")
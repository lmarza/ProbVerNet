import numpy as np; np.random.seed(2)
from scripts.Node import Node
import torch

class eProVe( ):

	def __init__( 
			self, network, input_domain, point_cloud, 
			split_node_heu="distr", split_pos_heu="distr", split_test_heu="none", max_depth=18
		  ):
		
		# Constructor paramiters
		self.network = network
		self.input_domain = np.array(input_domain)
		self.point_cloud = point_cloud
		self.max_depth = max_depth

		# Heuristic definition
		# --> ["rand", "size", "distr"]
		self.split_node_heu = split_node_heu
		# --> ["median", "mean", "distr"]
		self.split_pos_heu = split_pos_heu 
		# --> ["entropy", "none"]
		self.split_test_heu = split_test_heu

		error_msg = "Invalid heuristic, check NewProve.py"
		# Sanity check for valid heurisitc
		assert self.split_node_heu in ["rand", "size", "distr"]
		assert self.split_pos_heu in ["median", "mean", "distr"]
		assert self.split_test_heu in ["entropy", "none"], error_msg
		# Sanity check for consistency
		if self.split_node_heu == "distr" or self.split_pos_heu == "distr": 
			assert self.split_node_heu == "distr" and self.split_pos_heu == "distr", error_msg

		# Private config variables
		self._time_out_cycle = 25
		

	def verify( self ):

		root = Node( value=self.input_domain, network=self.network, split_node_heu=self.split_node_heu, split_pos_heu=self.split_pos_heu, split_test_heu=self.split_test_heu, max_depth=self.max_depth )

		frontier = [ root ]
		next_frontier = []
		safe_areas = []

		for depth in range( self._time_out_cycle ):

			# print( f"depth: {depth} (len-frontier {len(frontier)})" )

			# Itearate over the nodes in the frontier
			for node in frontier:

				# If the node is verified, add it to the safe areas list
				if node.get_probability( self.point_cloud ) == 1: 
					safe_areas.append( node )
					continue

				# If the node passes the test, split into the two children
				if node.expansion_test():
					child_1, child_2 = node.get_children()
					next_frontier.append( child_1 )
					next_frontier.append( child_2 )

			# The tree has been completely explored
			if frontier == []: break

			# Update the frontier with the new explored nodes
			frontier.clear()
			frontier = next_frontier.copy()
			next_frontier.clear()

		# Compute the underestimation of the safe areas and the safe rate
		safe_size = sum([ subarea.compute_area_size() for subarea in safe_areas ])
		total_size = root.compute_area_size()
		safe_rate = safe_size / total_size

		# Create a dictionary with the additional info
		info = { 
			'areas': [subarea.value for subarea in safe_areas], 
			'areas-object': [subarea for subarea in safe_areas], 
			'areas-number': len([subarea.value for subarea in safe_areas]),
	  		'depth-reached': depth
	  	}
		
		#
		return safe_rate, info


	def estimate_safe_rate( self ):

		"""
		Utility method for the estimation of the safe rate in the initial property.

		Returns:
		--------
			safe_rate : float
				estimation of the safe-rate computed with 1000000 points
		"""

		network_input = np.random.uniform(self.input_domain[:, 0], self.input_domain[:, 1], size=(1000000, self.input_domain.shape[0]))
		network_input = torch.from_numpy(network_input).float()
		network_output = self.network(network_input).detach().numpy()

		safe_rate = np.where(network_output > 0)[0].shape[0] / 1000000
		return safe_rate
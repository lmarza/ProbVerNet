import torch
import numpy as np

class Node:

	def __init__( self, value, network, split_node_heu, split_pos_heu, split_test_heu, max_depth=18, depth=0, parent=None ):
		
		# Parameters
		self.value = value
		self.parent = parent
		self.network = network
		self.depth = depth
		self.max_depth = max_depth

		# Heuristic definition
		self.split_node_heu = split_node_heu 
		self.split_pos_heu = split_pos_heu
		self.split_test_heu = split_test_heu 

		# Private variables
		self._probability = None
		self._children = [None, None]
		
		# Heuristic Variables
		self._propagated_median = None
		self._max_pivots = None
		self._min_pivots = None

		# Internal variables for debugging
		self.splitted_on = None


	def get_probability(self, point_cloud=3500):

		"""
		Public method that returns the probability of having a safe point in the current node, 
		this probability is calculated by sampling 'point_cloud' points

		Returns:
		--------
			probability : float
				the probability of having a safe point; 100% means that all the sampled points
				are safe
		"""

		
		# If the probability is already being calculated, just return
		if self._probability is not None:
			return self._probability

		# Generate a point cloud of size 'point cloud' and propagate it
		# through the network
		network_input = np.random.uniform(self.value[:, 0], self.value[:, 1], size=(point_cloud, self.value.shape[0]))
		network_input = torch.from_numpy(network_input).float()
		network_output = self.network(network_input).detach().numpy()

		# Compute the portion of the points that respects
		# the requirements; notice that 100% of the points mean that
		# the property is completely safe in the given area (lower bound greater than zero)
		indexes = np.where(network_output >= 0)[0]
		self._probability = indexes.shape[0] / point_cloud
		
		# These two special cases will never been splitted, however for heuristic
		# purposes we must return a placeholder for the heuristic variables, it 
		# will never be used for real, but it must be not null. Otherwise it will 
		# generate an error. Notice that the children (although never used), must
		# be created for the computation of the entropy condition for the splitting.
		# It won't be used anyway beacuse the probability 1 or 0 has higher priority.
		if self._probability == 1 or self._probability == 0: 
			self._propagated_median = [0, 0]
			self._max_pivots = [0 for _ in self.value]
			self._min_pivots = [0 for _ in self.value]
			return self._probability

		# Computing additional info for heuristic purposes
		max_points = np.max(network_input[indexes].numpy(), axis=0) 
		min_points = np.min(network_input[indexes].numpy(), axis=0)
		# 		-> Storing the information
		self._propagated_median = np.median( network_input[indexes], axis=0 )
		self._max_pivots = (max_points - self.value[:, 0]) / (self.value[:, 1] - self.value[:, 0])
		self._min_pivots = (min_points - self.value[:, 0]) / (self.value[:, 1] - self.value[:, 0])

		# Return the computed probability if not already returned
		return self._probability
		

	def get_children( self ):

		"""
		Public method that generates the two children nodes, typically called only when the 
		testing method returns True.

		Returns:
		--------
			childred : list
				an array of two elements with the children of the node
		"""

		# If the children are already generated, just return
		if self._children[0] is not None: return self._children

		# print( "prob", self.get_probability() )

		# Call the function that implements the heuristic for the choice
		# of the node to split
		node, pivot = self._chose_split()
		self.splitted_on = node

		# Create a copy of the current area before changing it for the childre;
		# NB: the copy function is necessary to avoid any change in the parent node
		value_1 = self.value.copy() 
		value_2 = self.value.copy() 

		# Change lower and upper bound of the children
		value_1[node][1] = value_2[node][0] = pivot

		# Call the Node class for the generation of the nodes
		self._children = [
			Node( value=value_1, network=self.network, depth=self.depth+1, parent=self, split_node_heu=self.split_node_heu, split_pos_heu=self.split_pos_heu, split_test_heu=self.split_test_heu, max_depth=self.max_depth ),
			Node( value=value_2, network=self.network, depth=self.depth+1, parent=self, split_node_heu=self.split_node_heu, split_pos_heu=self.split_pos_heu, split_test_heu=self.split_test_heu, max_depth=self.max_depth )
		]

		# Return the generated childred if not already returned
		return self._children
			

	def compute_area_size( self ):

		"""
		Public method that computes the size of the area represented with the current node.

		Returns:
		--------
			size : float
				the size of the area represented with the node
		"""
				
		# Compute the size of each side and return the product
		sides_sizes = [ side[1] - side[0] for side in self.value ]
		return np.prod(sides_sizes) 
	

	def _compute_entropy( self, p ):
		p = max(p, 0.0001)
		p = min(p, 0.9999)
		return -p * np.log2(p) - (1-p) * np.log2(1-p)


	def expansion_test( self ):


		if self._children[0] is None: self.get_children()

		# Notice that the case self.get_probability() == 1 has already been computed
		if self.get_probability() < 0.001: 
			return False
		elif self.depth > self.max_depth: 
			return False
		else: return True


	def _chose_split( self ):

		"""
		Private method that select the node on which performs the splitting, the implemented heuristic is to always select the node
		with the largest bound.

		Returns:
		--------
			distance_index : int
				index of the selected node (based on the heuristic)
		"""

		########
		## Heuristic Based on The Distribution
		########
		if self.split_node_heu == "distr" or self.split_pos_heu == "distr":
			test_flag = False

			# Condition 1
			if np.min(self._max_pivots, axis=0) < 0.9: 
				pivot = np.min(self._max_pivots, axis=0)
				node = np.argmin(self._max_pivots, axis=0)
				test_flag = True
				
			# Condition 2
			if np.max(self._min_pivots, axis=0) > 0.1: 
				pivot = np.max(self._min_pivots, axis=0)
				node = np.argmax(self._min_pivots, axis=0)
				test_flag = True

			#
			if test_flag: 
				pivot_value = (pivot*(self.value[node][1]-self.value[node][0]))+self.value[node][0]
				return node, pivot_value 
			else:
				# If there is no need to use the distr heurisitc use the standard
				distances = [ (el[1] - el[0]) for el in self.value ]
				node = np.argmax(distances)
				pivot_value = self._propagated_median[node]
				return node, pivot_value
		
		########
		## Heuristic for the Node Selection
		########
		if self.split_node_heu == "rand": 
			node = np.random.randint(0, self.value.shape[0]) 
		elif self.split_node_heu == "size":
			distances = [ (el[1] - el[0]) for el in self.value ]
			node = np.argmax(distances)
		else:
			print( f"Invalid Heurisitc Check... {self.split_node_heu}"); quit()

		########
		## Heuristic for the Split Position
		########
		if self.split_pos_heu == "mean":
			pivot_value = (self.value[node][0]+self.value[node][1])*0.5
		elif self.split_pos_heu == "median":
				pivot_value = self._propagated_median[node]
		else:
			print( f"Invalid Heurisitc Check... {self.split_pos_heu}"); quit()

		# 
		return node, pivot_value
	

class eProVe( ):

	def __init__( 
			self, network, input_domain, point_cloud, desired_coverage=75,
			split_node_heu="distr", split_pos_heu="distr", split_test_heu="none", max_depth=21
		  ):
				
		# Constructor paramiters
		self.network = network
		self.input_domain = input_domain
		self.point_cloud = point_cloud
		self.max_depth = max_depth
		self.desired_coverage = desired_coverage
		self.estimated_safe_rate = round(self.estimate_safe_rate()*100,2)

		# Heuristic definition
		# --> ["rand", "size", "distr"]
		self.split_node_heu = split_node_heu
		# --> ["median", "mean", "distr"]
		self.split_pos_heu = split_pos_heu 
		# --> ["entropy", "none"]
		self.split_test_heu = split_test_heu

		error_msg = "Invalid heuristic"
		# Sanity check for valid heurisitc
		assert self.split_node_heu in ["rand", "size", "distr"]
		assert self.split_pos_heu in ["median", "mean", "distr"]
		assert self.split_test_heu in ["entropy", "none"], error_msg
		# Sanity check for consistency
		if self.split_node_heu == "distr" or self.split_pos_heu == "distr": 
			assert self.split_node_heu == "distr" and self.split_pos_heu == "distr", error_msg

		# Private config variables
		self._time_out_cycle = 18
		

	def verify(self):

		root = Node(value=self.input_domain, network=self.network, split_node_heu=self.split_node_heu, split_pos_heu=self.split_pos_heu, split_test_heu=self.split_test_heu, max_depth=self.max_depth)

		frontier = [root]
		next_frontier = []
		safe_areas = []
		total_size = root.compute_area_size()

		for depth in range(self.max_depth):
	
			# Itearate over the nodes in the frontier
			for node in frontier:

				# If the node is verified, add it to the safe areas list
				if node.get_probability(self.point_cloud) == 1.0: 
					safe_areas.append(node)
					continue

				# If the node passes the test, split into the two children
				if node.expansion_test():
					child_1, child_2 = node.get_children()
					next_frontier.append(child_1)
					next_frontier.append(child_2)

			# The tree has been completely explored
			if frontier == []: break

			# Update the frontier with the new explored nodes
			frontier.clear()
			frontier = next_frontier.copy()
			next_frontier.clear()

			# Compute the underestimation of the safe areas and the safe rate
			safe_size = sum([ subarea.compute_area_size() for subarea in safe_areas ])
			safe_rate = round((safe_size / total_size)*100, 4)
			print(f"Depth: {depth+1} (len-frontier {len(frontier)})" )
			print("\t# of safe regions: ", len(safe_areas))
			print(f"\tcoverage: {(safe_rate/self.estimated_safe_rate) * 100} %\n")
			if (safe_rate/self.estimated_safe_rate) * 100 >= self.desired_coverage: 
				break


		
		# Generate random test points in the input domain
		network_input = np.random.uniform(self.input_domain[:, 0], self.input_domain[:, 1],size=(100_000, self.input_domain.shape[0]))
		network_input_tensor = torch.from_numpy(network_input).float()

		# Get the network output: True where output < 0 (i.e., unsafe points)
		network_output_mask = self.network(network_input_tensor) < 0
		network_output_mask = network_output_mask.detach().numpy().flatten()

		# Filter unsafe inputs
		unsafe_points = network_input[network_output_mask]

		# Initialize percentage
		percentage_unsafe_included = 0.0
		included_mask = np.zeros(len(unsafe_points), dtype=bool)

		for area in safe_areas:
			bounds = area.value  # shape: (n_dims, 2)
			lower = bounds[:, 0]
			upper = bounds[:, 1]

			# Check which unsafe points are within this area
			in_area = np.all((unsafe_points >= lower) & (unsafe_points <= upper), axis=1)
			included_mask |= in_area  # mark as included if in any area

		percentage_unsafe_included = included_mask.sum() / self.point_cloud * 100


		

		# Create a dictionary with the additional info
		info = { 
			'areas': [subarea.value for subarea in safe_areas], 
			'areas-object': [subarea for subarea in safe_areas], 
			'areas-number': len([subarea.value for subarea in safe_areas]),
			'coverage': (safe_rate/self.estimated_safe_rate)*100,
			'error': percentage_unsafe_included,
	  		'depth-reached': depth
	  	}
		
		#
		return self.estimated_safe_rate, info


	def estimate_safe_rate( self ):

		"""
		Utility method for the estimation of the safe rate in the initial property.

		Returns:
		--------
			safe_rate : float
				estimation of the safe-rate computed with 1000000 points
		"""

		network_input = np.random.uniform(self.input_domain[:, 0], self.input_domain[:, 1], size=(1_000_000, self.input_domain.shape[0]))
		network_input = torch.from_numpy(network_input).float()
		network_output = self.network(network_input).detach().numpy()

		safe_rate = np.where(network_output >= 0)[0].shape[0] / 1_000_000
		return safe_rate
	

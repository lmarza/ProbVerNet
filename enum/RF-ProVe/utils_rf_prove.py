from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import torch 
import numpy as np
import torch.nn as nn

CYAN_COL = '\033[96m'
BLUE_COL = '\033[94m'
RED_COL = '\033[91m'
GREEN_COL = '\033[92m'
YELLOW_COL = '\033[93m'
RESET_COL = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'




# Wrap the model to override forward
class BinaryDecisionWrapperCartPole(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.where(out[:, 0] >= out[:, 1], torch.tensor(1), torch.tensor(-1))
    
class BinaryDecisionWrapperLunarLander(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.where(torch.argmax(out, dim=1) == 1, torch.tensor(1), torch.tensor(-1))
    
class BinaryDecisionWrapperDubinsrejoin(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)  # shape: [batch_size, 8]

        pred_R = out[:, :4]
        pred_T = out[:, 4:]

        pred_label_R = pred_R.argmax(dim=1)
        pred_label_T = pred_T.argmax(dim=1)

        # Check if argmax is 0 (label for R) and 0 (label_T == 4 → index 0 in pred_T)
        mask = (pred_label_R == 0) & (pred_label_T == 0)

        result = torch.where(mask, torch.tensor(1), torch.tensor(-1))
        return result

class BinaryDecisionWrapperVCAS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.where(torch.argmax(out, dim=1) == 0, torch.tensor(1), torch.tensor(-1))


class SimpleNN(nn.Module):
    def __init__(self, input_size=2):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)  
        self.layer2 = nn.Linear(32, 32)  
        self.layer3 = nn.Linear(32, 32) 
        self.output_layer = nn.Linear(32, 1) 

    def forward(self, x):
        x = torch.relu(self.layer1(x))  
        x = torch.relu(self.layer2(x)) 
        x = torch.relu(self.layer3(x)) 
        x = self.output_layer(x) 
        return x
    
class AcasNetP2(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(AcasNetP2, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 50)
		self.fc2 = torch.nn.Linear(50, 50)
		self.fc3 = torch.nn.Linear(50, 50)
		self.fc4 = torch.nn.Linear(50, 50)
		self.fc5 = torch.nn.Linear(50, 50)
		self.fc6 = torch.nn.Linear(50, 50)
		self.fc7 = torch.nn.Linear(50, output_size)

	def forward(self, x, real_output=False):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = torch.torch.nn.functional.relu(self.fc3(x))
		x = torch.torch.nn.functional.relu(self.fc4(x))
		x = torch.torch.nn.functional.relu(self.fc5(x))
		x = torch.torch.nn.functional.relu(self.fc6(x))
		x = self.fc7(x)

		if real_output: return x
		
		res = [1 if el else -1 for el in torch.argmax(x, dim=1).numpy() != 0]
		return torch.tensor(res)  
		

# Torch class for the TurtleBot test (Property 'front')
class TurtleNetFront(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(TurtleNetFront, self).__init__()
		self.fc1 = torch.nn.Linear(input_size, 64)
		self.fc2 = torch.nn.Linear(64, 64)
		self.fc3 = torch.nn.Linear(64, output_size)

	def forward(self, x, real_output=False):
		x = torch.torch.nn.functional.relu(self.fc1(x))
		x = torch.torch.nn.functional.relu(self.fc2(x))
		x = torch.torch.nn.functional.softmax(self.fc3(x))

		if real_output: return x
		
		res = [1 if el else -1 for el in torch.argmax(x, dim=1).numpy() != 4]
		return torch.tensor( res )  
		


def exact_remove_contained_regions(current_pure_regions, filtered_pure_regions):
    
    if len(filtered_pure_regions) == 0:
        return current_pure_regions.copy()
    
    result = filtered_pure_regions.copy()

    for pure_region in current_pure_regions:
        is_contained = False
        for filtered_pure_region in filtered_pure_regions:
            if np.all(pure_region[:, 0] >= filtered_pure_region[:, 0]) and \
                np.all(pure_region[:, 1] <= filtered_pure_region[:, 1]):
                is_contained = True
                break
        if not is_contained:
            result.append(pure_region)
    
    return result

    

def get_pure_leaf_rectangles(tree, input_domain):
    """
    Extract hyperrectangles corresponding to pure leaves with class 1 from a decision tree.

    :param tree: Trained decision tree classifier.
    :param input_domain: np.array of shape (n_dims, 2), defining the min/max range for each dimension.
    :return: List of n-dimensional hyperrectangles defining the decision regions for pure leaves.
    """
    tree_ = tree.tree_
    n_dims = input_domain.shape[0]  # Extract number of dimensions dynamically

    # Initialize bounds as a dictionary {node_index: [[min1, max1], ..., [minN, maxN]]}
    node_bounds = {0: [list(input_domain[i]) for i in range(n_dims)]}

    pure_leaf_rects = []

    def traverse(node, bounds):
        """
        Recursively traverse the tree and track feature bounds.
        """
        if tree_.children_left[node] == -1 and tree_.children_right[node] == -1:  # Leaf node
            value = tree_.value[node][0]
            if np.sum(value) == value[1]:  # All samples in this leaf are class 1
                pure_leaf_rects.append(bounds)
            return

        feature = tree_.feature[node]  # Feature index that splits this node
        threshold = tree_.threshold[node]

        # Copy bounds for left and right children
        left_bounds = [b[:] for b in bounds]   # Deep copy for left
        right_bounds = [b[:] for b in bounds]  # Deep copy for right

        # Update bounds based on the split feature
        left_bounds[feature][1] = threshold   # Shrink upper bound for left child
        right_bounds[feature][0] = threshold  # Shrink lower bound for right child

        traverse(tree_.children_left[node], left_bounds)
        traverse(tree_.children_right[node], right_bounds)


    traverse(0, node_bounds[0])  # Start from root node
    return pure_leaf_rects


def get_test_inputs(input_domain, model, digits_to_consider, sample_size=100_000):
    # Sample inputs
    inputs = np.random.uniform(input_domain[:, 0], input_domain[:, 1], size=(sample_size, input_domain.shape[0]))
    # round the inputs to digits_to_consider decimal places
    if digits_to_consider > 0:
        inputs = np.round(inputs, digits_to_consider)
    input_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Get model outputs (vectorized computation)
    with torch.no_grad():
        outputs = model(input_tensor).squeeze().numpy() 
        # Convert outputs to binary labels
        outputs = (outputs >= 0)

    result = {
        'unsafe_inputs': inputs[~outputs],
        'safe_inputs': inputs[outputs]
    }
    
    return result



def is_inside(points, hyperrectangles):
    """Check if a point is inside any of the given hyperrectangles."""
    points = np.expand_dims(points, axis=1)  # Shape (M, 1, n)
    hyperrectangles = np.array(hyperrectangles)  # Shape (R, n, 2), where R is number of hyperrectangles

    mins = hyperrectangles[:, :, 0]  # Shape (R, n)
    maxs = hyperrectangles[:, :, 1]  # Shape (R, n)

    inside = np.all((points >= mins) & (points <= maxs), axis=2)  # Shape (M, R)
    return np.any(inside, axis=1) 

def check_coverage(hyperrectangles, input_to_test, check_unsafe=True, check_training_samples=True):

    # input_to_test is a dictionary that has both safe and unsafe inputs. First update input_to_test removing the safe input correctly mapped into hyperrectangles
    safe_input_to_test = input_to_test['safe_inputs']
    inside_mask = is_inside(safe_input_to_test, hyperrectangles)
    # Remove safe inputs that are inside any hyperrectangle
    input_to_test['safe_inputs'] = safe_input_to_test[~inside_mask]

    if check_unsafe:
        unsafe_input_to_test = input_to_test['unsafe_inputs']
        inside_mask = is_inside(unsafe_input_to_test, hyperrectangles)
        # Remove safe inputs that are inside any hyperrectangle
        input_to_test['unsafe_inputs'] = unsafe_input_to_test[~inside_mask]

    if check_training_samples:
        training_samples = input_to_test['training_samples']
        inside_mask = is_inside(training_samples, hyperrectangles)
        # Remove training samples that are inside any hyperrectangle
        input_to_test['training_samples'] = training_samples[~inside_mask]

    return input_to_test

def estimate_safe_volume(hyperrectangles, inputs, input_domain):
    """
    Compute the volume of safe regions using Monte Carlo sampling.
    
    Parameters:
    - hyperrectangles: List of hyperrectangles, each in the form [[min_1, max_1], ..., [min_n, max_n]]
    - sample_points: Numpy array of sampled points (M x n)
    - total_volume: Volume of the entire input space

    Returns:
    - Estimated volume of the safe region
    """

    # Compute the volume of the entire input space
    total_volume = np.prod(input_domain[:, 1] - input_domain[:, 0])

    count_inside = np.sum(is_inside(inputs, hyperrectangles))
    return total_volume * (count_inside / len(inputs)) 



def compute_estimate_rate(model, input_domain, digits_to_consider=3, safe_rate=True, sample_size=1_000_000):

    # Sample inputs
    inputs = np.random.uniform(input_domain[:, 0], input_domain[:, 1], size=(sample_size, input_domain.shape[0]))

    if digits_to_consider > 0:
        inputs = np.round(inputs, digits_to_consider)

    input_tensor = torch.tensor(inputs, dtype=torch.float32)
   

    # Get model outputs (vectorized computation)
    with torch.no_grad():
        outputs = model(input_tensor).squeeze().numpy()
        # Convert outputs to binary labels
        outputs = (outputs >= 0)

    if safe_rate:
        estimate_rate = round((np.sum(outputs) / sample_size) * 100, 2)
    else:
        estimate_rate = round((1 - np.sum(outputs) / sample_size) * 100, 2)

    return estimate_rate


def train_rf(model, input_domain, param_rf, safe_regions=True):

    sample_to_train, n_decision_trees, max_depth, heuristic_rf, seed, digits_to_consider = param_rf

    # Sample inputs
    inputs = np.random.uniform(input_domain[:, 0], input_domain[:, 1], size=(sample_to_train, input_domain.shape[0]))
    # round the inputs to digits_to_consider decimal places
    if digits_to_consider > 0:
        inputs = np.round(inputs, digits_to_consider)

    input_tensor = torch.tensor(inputs, dtype=torch.float32)

    # Get model outputs (vectorized computation)
    with torch.no_grad():
        outputs = model(input_tensor).squeeze().numpy() 
        # Convert outputs to binary labels
        outputs = (outputs >= 0)


    
    rf = RandomForestClassifier(n_estimators=n_decision_trees, criterion=heuristic_rf, max_depth=max_depth, random_state=seed, n_jobs=-1)
    rf.fit(inputs, outputs)

    # collect only inputs that prod

    return rf, inputs[outputs]

def verify(model, input_domain, param_rf, safe_regions=True, save_regions=False, verbose=True, wilks_filtering=True, wilks_samples=3500, sample_size_evaluation=100_000, desired_coverage=99, digits_to_consider=0, compute_min_side=False):

    # Compute estimated safe regions
    estimated_rate = compute_estimate_rate(model, input_domain, digits_to_consider=param_rf[-1], safe_rate=safe_regions, sample_size=1_000_000)

    if estimated_rate in [0.0, 100.0]:
        print(f"All domain is {'safe' if estimated_rate == 100.0 else 'unsafe'}")
        return
       
    # Train Random Forest
    print("Training random forest...")
    rf, training_samples = train_rf(model, input_domain, param_rf, safe_regions)
    print("Finished training random forest...\nComputing set of positive polytopes...\n")

    # retrieve valid regions
    filtered_pure_regions = []
    current_pure_regions = []
    all_leaves = 0
    depth = []
    
    total_safe_volume = 0
    input_dict = get_test_inputs(input_domain, model, digits_to_consider, sample_size=sample_size_evaluation)
    n_original_safe_inputs = len(input_dict['safe_inputs'])
    n_original_unsafe_inputs = len(input_dict['unsafe_inputs'])
    input_dict['training_samples'] = training_samples
    n_training_samples = len(input_dict['training_samples'])
    
    for i, tree in enumerate(rf):
        
        rects = get_pure_leaf_rectangles(tree, input_domain)
      
        all_leaves += len(rects)
        depth.append(tree.tree_.max_depth)
    
        for rect in rects:

            if digits_to_consider > 0:
                rect = np.round(np.array(rect), digits_to_consider)
            else:
                rect = np.array(rect)
            

            if wilks_filtering:
                # Wilks check
                random_sample = np.random.uniform(rect[:, 0], rect[:, 1], size=(wilks_samples, rect.shape[0]))
                random_sample = torch.tensor(random_sample, dtype=torch.float32)
                if digits_to_consider > 0:
                    random_sample = torch.tensor(np.round(random_sample, digits_to_consider), dtype=torch.float32)
                with torch.no_grad():
                    output = model(random_sample).squeeze().numpy()
                    if np.sum(output >= 0) == wilks_samples:  # Fully safe
                        current_pure_regions.append(rect)
            else:
                current_pure_regions.append(rect)

        
        # now remove the regions that are completely contained in another region
        filtered_pure_regions = exact_remove_contained_regions(current_pure_regions, filtered_pure_regions)
        current_pure_regions = []
        percentage_safe_covered = 0
        percentage_unsafe_included = 0

        # test coverage
        if len(filtered_pure_regions) > 0:
            # check how many inputs are mapped in the filtered leaves, i.e., in the probabilistically safe regions
            input_dict = check_coverage(filtered_pure_regions, input_dict)
            percentage_safe_covered = round((n_original_safe_inputs - len(input_dict['safe_inputs']))/n_original_safe_inputs * 100,2)
            percentage_unsafe_included = round((n_original_unsafe_inputs - len(input_dict['unsafe_inputs']))/sample_size_evaluation * 100,2)
            
            if verbose:
                print("\nTree:", i+1, "/", len(rf))
                print("\tNumber of polytopes: ", len(filtered_pure_regions))
                print(f"\t#Safe inputs to be covered: {len(input_dict['safe_inputs'])}/{n_original_safe_inputs} |  #Unsafe inputs: {len(input_dict['unsafe_inputs'])}/{n_original_unsafe_inputs} | #Training samples to be covered: {len(input_dict['training_samples'])}")
                print(f"\tPercentage of positive test set covered: {percentage_safe_covered}%")
                print(f"\tPercentage of non-positive inputs in collected polytopes: {percentage_unsafe_included}%")
                #print("\tPercentage of safe training samples covered: ", round((n_training_samples - len(input_dict['training_samples']))/n_training_samples * 100, 2), "%")

            
            if percentage_safe_covered > desired_coverage:
                break


    safe_points_correctly_mapped = n_original_safe_inputs - len(input_dict['safe_inputs'])    
    total_safe_volume = round((safe_points_correctly_mapped / sample_size_evaluation) * 100, 2)
    estimate_safe_rate = round((n_original_safe_inputs / sample_size_evaluation) * 100, 2)
    total_error = round((abs(estimate_safe_rate - total_safe_volume) / estimate_safe_rate) * 100, 4)
    #training_samples_covered = round((n_training_samples - len(input_dict['training_samples']))/ n_training_samples * 100, 2)

    # safe valid regions
    if save_regions: np.save(f"regions/safe_regions_model_{input_domain.shape[0]}_{round(estimated_rate/100, 2)}.npy", filtered_pure_regions)

    # check the minimum size of side stored in filtered_pure_regions.
    min_side = None
    if compute_min_side and len(filtered_pure_regions) > 0:
        min_side = min(np.min(rect[:, 1] - rect[:, 0]) for rect in filtered_pure_regions)


    return estimated_rate, len(filtered_pure_regions), 100 - total_error, percentage_unsafe_included, min_side
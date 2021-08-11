
# functions that use all classifiers
from .for_all import try_all_classifiers

# functions for each classifiers
from .decision_tree import decision_tree
from .k_nearest_neighbors import k_nearest_neighbors
from .logistic_regression import logistic_regression
from .random_forest import random_forest
from .support_vectors import support_vectors
from .team import team
from .deep_neural_network import deep_neural_network

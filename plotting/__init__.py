
# Functions to assist in plotting from .utilities subpackage
from .utilities import _get_colors, _get_markers, _get_lines_styles
from .utilities import calculate_figure_dims

# Plotting points functions from .points subpackage
from .points import plot_points

# Plotting histograms functions from .histograms subpackage
from .histograms import plot_PCA_features_importances
from .histograms import plot_data_columns_counts
from .histograms import plot_data_target_dependencies

# Plotting matrices functions from .matrices subpackage
from .matrices import plot_confusion_matrix

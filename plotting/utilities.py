import numpy as np


def _get_markers(count: int = 1, random: bool = False):
    """
    Get list of markers which can be used during plotting
    charts or diagrams in matplotlib package.

    Parameters
    ----------
    count : int {default: 1}
        Number of needed marker. Has to be >=1.

    random : bool {default: False}
        Are markers have to be given randomly.

    References
    ----------
    [1] https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    Returns
    -------
    markers
        List of markers ready to use in matplotlib package.
    """
    markers = np.array(['o', 'v', '^', '<', '>',
                        '1', '2', '3', '4', '8',
                        's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'P', 'X'])

    if random:
        np.random.shuffle(markers)

    return markers[:count]


def _get_colors(count: int = 1, kind: str = "float", random: bool = False):
    """
    Get list of colors which can be used during plotting
    charts or diagrams in matplotlib package.

    Parameters
    ----------
    count : int {default: 1}
        Number of needed marker. Has to be >=1.

    kind : str {default: "float"}
        Data type (in string) whose colors will be returned.

    random : bool {default: False}
        Are colors have to be given randomly.

    Returns
    -------
    colors
        List of colors ready to use in matplotlib package
    """
    colors_str = ["blue", "orange", "green", "red", "purple", "brown",
                  "pink", "gray", "olive", "cyan", "rosybrown", "goldenrod",
                  "aquamarine", "darkslategrey", "skyblue", "magenta",
                  "indigo", "crimson"]

    if kind in ["int", "float"]:
        colors = np.linspace(0, 1, count)
    elif kind in ["str"]:
        colors = colors_str[:count]
    else:
        colors = None

    if random and colors is not None:
        np.random.shuffle(colors)

    return colors


def _get_lines_styles(count: int = 1):
    """
    Get list of line styles which can be used during
    plotting charts or diagrams in matplotlib package.

    Parameters
    ----------
    count : int {default: 1}
        Number of needed styles. Has to be >=1.

    Returns
    -------
    colors
        List of line styles ready to use in matplotlib package.
    """
    styles = {'-': '_draw_solid',
              '--': '_draw_dashed',
              '-.': '_draw_dash_dot',
              ':': '_draw_dotted'}

    multiple = np.ceil(count/len(styles)).astype(np.int32)
    styles_list = []
    for _ in range(multiple):
        for style in styles:
            styles_list.append(style)

    return styles_list[:count]


def calculate_figure_dims(sub_charts: int):
    """
    Calculate how many rows and columns has to be the best looking figure.

    Parameters
    ----------
    sub_charts : int
        Number >= 0 defines how many sub charts are on figure.

    Returns
    -------
    (rows, cols)
        Tuple of rows and columns numbers of best looking figure.
    """
    if sub_charts < 0:
        sub_charts = 0

    fig_cols = np.floor(np.sqrt(sub_charts)).astype(np.int32)
    fig_rows = np.ceil(sub_charts / fig_cols).astype(np.int32)

    return fig_rows, fig_cols

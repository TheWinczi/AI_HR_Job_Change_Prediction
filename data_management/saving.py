import pandas as pd


def save_data(data: pd.DataFrame,
              out_file_path: str):
    """
    Save data into given path as .csv file.

    Parameters
    ----------
    data : DataFrame
        Data to save as DataFrame object.

    out_file_path : str
        Path to out file.
    """
    try:
        data.to_csv(out_file_path, index=False)
    except IOError:
        print('Saving file with processed data failed')
import pandas as pd


#
def load_data(file_name, sep=','):
    """
    A data loader function for csv files
    :param file_name: str, the path to the file
    :param sep: str, the separator, defoult ','

    """

    data = pd.read_csv(file_name, sep=sep)

    return data

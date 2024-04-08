import numpy as np

# Utils
def discretize(data, nbins=10, var_type=None):
    """
    Discretize the numeric value of the data table

    :param data: numpy.array containing all the data
    :param var_type: list of int indicating the type of each column of data
    :param nbins: number of bins to create
    :return: numpy.array with discretized values
    """
    if type(var_type) == type(None):
        var_type = list(np.full(data.shape[1], "Numeric"))
    output_arr = np.zeros(data.shape)
    for col_index, col_type in enumerate(var_type):
        if col_type == "Numeric":
            bins = np.linspace(np.nanmin(data[:, col_index]), np.nanmax(data[:, col_index]), nbins)
            output_arr[:, col_index] = np.digitize(data[:, col_index], bins) - 1
        elif col_type == "Categorical":
            _, inverse = np.unique(data[:,col_index], return_inverse = True)
            output_arr[:, col_index] = inverse

    return output_arr

def infer_var_type(data):
    """
    Infer the type of each column of the data matrix

    :param data: numpy.array containing all the data
    :return: list of str indicating the type of each column of data from Numeric and Categorical
    """
    var_type = []
    for col in data.T:
        if np.unique(col).shape[0] < data.shape[0] / 10:
            var_type.append("Categorical")
        else:
            var_type.append("Numeric")
    return var_type
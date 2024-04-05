import numpy as np

def quality_evaluation_bicluster(bicluster, data, var_type=None, **kwargs):
    """
    Evaluate the quality of one specific bicluster using the heterogeneous intra-
    bicluster variance a weighted average of the average numeric variance of the numeric variables
    and the average categorical frequency for the catgeorical variables

    :param bicluster: iterable,
    contains 2 numpy.ndarray masks for the rows and the columns of the bicluster

    :param data: numpy.ndarray,
    input data matrix with all variables

    :param var_type: numpy.ndarray,
    ordered array containing the type of each column of the matrix data
    
    :return: float,
    quality of the bicluster
    """
    if type(var_type) == type(None):
        var_type = np.full(var_cols.shape[0], "Numeric")
    bic_matrix = data[np.ix_(bicluster[0], bicluster[1])]
    bic_var_type = var_type[bicluster[1]]

    numeric_columns_bic = np.where(bic_var_type == "Numeric")[0]
    cat_columns_bic = np.where(
        (bic_var_type == "Categorical") | (bic_var_type == "Binary")
    )[0]
    acf_list = np.zeros(cat_columns_bic.shape[0])
    if numeric_columns_bic.shape[0] + cat_columns_bic.shape[0] != bic_var_type.shape[0]:
        raise Exception(
            "At least one entry of var_type is incorrect"
        )  # Change exception type

    if "var_cols" in kwargs:
        var_cols = kwargs["var_cols"]
    else:
        var_cols = np.zeros(data.shape[1])
        var_cols[bicluster[1]][numeric_columns_bic] = np.var(
            data[:, numeric_columns_bic], axis=0
        )
    bic_variance = np.var(bic_matrix[:, numeric_columns_bic], axis=0)
    anv_list = bic_variance / var_cols[bicluster[1]][numeric_columns_bic]

    for i_col in range(cat_columns_bic.shape[0]):
        _, counts = np.unique(bic_matrix[:, cat_columns_bic[i_col]], return_counts=True)
        max_occurences = np.max(counts)
        acf_list[i_col] = 1 - (max_occurences / bic_matrix.shape[0])
    quality = (np.sum(anv_list) / max(anv_list.shape[0], 1)) + (
        np.sum(acf_list) / max(acf_list.shape[0], 1)
    )
    return quality


def quality_evaluation_biclusters(biclusters, data, var_type):
    """
    Evaluate the quality of every biclusters using the heterogeneous intra-
    bicluster variance a weighted average of the average numeric variance of the numeric variables
    and the average categorical frequency for the catgeorical variables

    :param biclusters: iterable,
    iterable of tuples that contains 2 numpy.ndarray masks for the rows and the columns of one individual bicluster

    :param data: numpy.ndarray,
    input data matrix with all variables

    :param var_type: numpy.ndarray,
    ordered array containing the type of each column of the matrix data
    
    :return: float,
    quality of the biclusters
    """
    if type(var_type) == str and var_type == "Numeric":
        var_type = np.full(data.shape[1], "Numeric")
    quality = np.zeros(len(biclusters))
    numeric_columns = np.where(var_type == "Numeric")[0]
    var_cols = np.zeros(data.shape[1])
    var_cols[numeric_columns] = np.var(data[:, numeric_columns], axis=0)
    for i in range(len(biclusters)):
        quality[i] = quality_evaluation_bicluster(
            biclusters[i], data, var_type, var_cols=var_cols
        )
    return quality

def sizes(biclusters):
    sizes = np.zeros(len(biclusters))
    for i in range(len(biclusters)):
        sizes[i] = sum(biclusters[i][0]) * sum(biclusters[i][1])
    return sizes

def score_biclusters(biclusters, data, var_type = "Numeric", lambda_hat = .5):
    q = quality_evaluation_biclusters(biclusters, data, var_type)
    q = 1 - (q / max(max(q), 1))
    s = sizes(biclusters)
    s = s / max(s)
    final_score = lambda_hat * s + (1 - lambda_hat) * q 
    return(final_score)

def L2_score_biclusters(biclusters, data, var_type = "Numeric", lambda_hat = .5):
    q = quality_evaluation_biclusters(biclusters, data, var_type)
    q = 1 - (q / max(max(q), 1))
    s = sizes(biclusters)
    s = s / max(s)
    final_score = lambda_hat * s**2 + (1 - lambda_hat) * q**2
    return(final_score)
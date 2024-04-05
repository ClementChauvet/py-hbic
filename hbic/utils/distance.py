import numpy as np

def distance(bic1, bic2):
    """
    Compute the distance between 2 biclusters by computing their overlaps
    
    :param bic1: tuple,
    tuple of 2 numpy.ndarray masks indicating the rows and columns of the bicluster
     
    :param bic2: tuple,
    tuple of 2 numpy.ndarray masks indicating the rows and columns of the bicluster
    
    :return: float,
    distance between the 2 biclusters
    """
    rows_intersection = bic1[0] * bic2[0]
    cols_intersection = bic1[1] * bic2[1]
    row_dist_bic1 = (sum(bic1[0]) - sum(rows_intersection)) / sum(bic1[0])
    col_dist_bic1 = (sum(bic1[1]) - sum(cols_intersection)) / sum(bic1[1])
    row_dist_bic2 = (sum(bic2[0]) - sum(rows_intersection)) / sum(bic2[0])
    col_dist_bic2 = (sum(bic2[1]) - sum(cols_intersection)) / sum(bic2[1])
    dist = row_dist_bic1 * col_dist_bic1 + row_dist_bic2 *  col_dist_bic2
    return dist

def distance_matrix(biclusters):
    """
    Compute the distance between every biclusters present in the list by scoring their overlaps
    
    :param biclusters: iterable,
    iterable of tuple of 2 numpy.ndarray masks indicating the rows and columns of the bicluster
     
    :return: numpy.ndarray,
    distance matrix where i,j represent the distance between bicluster i and bicluster j of the list
    """
    dist_matrix = np.zeros((len(biclusters), len(biclusters)))
    for i in range(len(biclusters)):
        for j in range(i + 1, len(biclusters)):
            dist = distance(biclusters[i], biclusters[j])
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
    return dist_matrix


def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<=costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
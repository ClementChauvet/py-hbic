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
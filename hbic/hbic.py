import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from .utils import distance, discretization, quality

class Hbic:
    def __init__(
        self,
        nbins=10,
        min_rows=(2, 0.03),
        min_cols=(2, 0.03),
        n_clusters="auto",
        reduction=None,
        verbose=False,
    ):
        """
        Initialize an object of class Hbic

        :param min_rows:  int, float or tuple of (int, float), default = (2, .03)
        select the minimum number of rows of a bicluster

            If an int is passed, specifies an absolute minimum number of rows

            If a float is passed, specifies the minimum proportion of rows

            If an iterable of format (n_abs, n_prop) is passed, the minimum number
            of rows will be computed using the formula max(n_abs, n_props * n_rows)


        :param min_cols: int, float or tuple of (int, float), default = (2, .03)
        select the minimum number of cols of a bicluster

            If an int is passed, specifies an absolute minimum number of cols

            If a float is passed, specifies the minimum proportion of cols

            If a tuple of format (n_abs, n_prop) is passed, the minimum number
            of rows will be computed using the formula max(n_abs, n_props * n_cols)


        :param n_clusters: int, default = "auto"
        Define the maximum number of bicluster to return. If an int is passed,
        a reduction parameter must also be passed

        :param reduction: str, default = None
        if passed the algorithm will reduce the number of biclusters found to obtain n_clusters
        a n_clusters parameter must also be passed

            'merging' : after an initial phase of bicluster detection, the algorithm will merge
            the most similar biclusters to obtain n_clusters

            'selection' : after an initial phase of bicluster detection, the algorithm will generate
            a dendogram of the biclusters following WARD procedure, the algorithm will then
            cut the dendogram according to n_cluster and the clusters in each branches will be merged

            'tree_selection' : after an initial phase of bicluster detection, the algorithm will generate
            a dendogram of the biclusters following WARD procedure, the algorithm will then
            cut the dendogram according to n_cluster and the biggest clusters of each branch will be selected
            
        :param verbose: bool, default = False
        Print out information during the process

        """
        # TODO : Revoir param reduction description

        if (
            isinstance(min_rows, Iterable)
            and isinstance(min_rows[0], int)
            and isinstance(min_rows[1], float)
        ):
            self.min_rows_abs = min_rows[0]
            self.min_rows_prop = min_rows[1]
        elif isinstance(min_rows, int):
            self.min_rows_abs = min_rows
            self.min_rows_prop = 0
        elif isinstance(min_rows, float):
            self.min_rows_abs = 0
            self.min_rows_prop = min_rows
        else:
            raise TypeError(
                "'min_rows' only allows iterable consisting of instances of int and float or int or float and not "+ str(type(min_rows))
            )

        if (
            isinstance(min_cols, Iterable)
            and isinstance(min_cols[0], int)
            and isinstance(min_cols[1], float)
        ):
            self.min_cols_abs = min_cols[0]
            self.min_cols_prop = min_cols[1]
        elif isinstance(min_cols, int):
            self.min_cols_abs = min_cols
            self.min_cols_prop = 0
        elif isinstance(min_cols, float):
            self.min_cols_abs = 0
            self.min_cols_prop = min_cols
        else:
            raise TypeError(
                "'min_cols' only allows int or float or iterable consisting of instances of int and float and not "+ str(type(min_cols))
            )

        self.n_clusters = n_clusters
        self.reduction = reduction
        self.verbose = verbose
        self.nbins = nbins

    def _find_best_column(self, arr_discretized, unclustered_columns):
        """
        Internal function to determine the best column to add to the bicluster.
        Return the value with the highest count in any column and the index of said column

        :param arr_discretized: numpy.ndarray,
        Filtered data where only the rows assigned to the bicluster remains

        :param unclustered_columns:iterable,
        Containing the columns that are not yet part of the bicluster


        :return: tuple of two int,
        first the value with the highest count in any column and the index of said column
        """

        # If the number of rows did not change when adding the previous column
        # we don't need to run the previous computations we just discard the column
        if self.check:
            self.iteration_counts = np.zeros(unclustered_columns.shape)
            self.iteration_values = np.zeros(unclustered_columns.shape)
            for i in range(len(unclustered_columns)):
                values, counts = np.unique(
                    arr_discretized[:, unclustered_columns[i]], return_counts=True
                )
                max_index = np.argmax(counts)
                self.iteration_counts[i] = counts[max_index]
                self.iteration_values[i] = values[max_index]

        max_col = np.argmax(self.iteration_counts)
        value, column = self.iteration_values[max_col], unclustered_columns[max_col]
        if arr_discretized.shape[0] != self.iteration_counts[max_col]:
            self.check = True
        else:
            self.check = False
            self.iteration_values = np.delete(self.iteration_values, max_col)
            self.iteration_counts = np.delete(self.iteration_counts, max_col)
        return value, column

    def _add_column(self, cols_ids_bic, rows_ids_bic, column, rows):
        """
        Internal function to add a column to an existing bicluster and remove rows
        that don't correspond anymore

        :param cols_ids_bic: 1D np.ndarray
        mask of columns that represent the bicluster

        :param rows_ids_bic: 1D np.ndarray
        mask of rows that represent the bicluster

        :param column: int
        index of the column to add

        :param rows: 1D numpy.ndarray
        mask of the rows in column that shares the same value
        """
        cols_ids_bic[column] = True
        for i in range(rows_ids_bic.shape[0]):
            if rows[i]:
                continue
            rows_ids_bic[i] = False

    def _remove_repeated_bic(self):
        """
        Internal function to remove biclusters that are repeated in self.biclusters
        """
        i = 0
        while i < len(self.biclusters) - 1:
            todelete = []
            for j in range(i + 1, len(self.biclusters)):
                rows = (self.biclusters[i][0] == self.biclusters[j][0]).all()
                if not rows:
                    continue
                columns = (self.biclusters[i][1] == self.biclusters[j][1]).all()
                if columns:
                    todelete.append(j)
            i += 1
            todelete.sort(reverse=True)
            for j in todelete:
                self.biclusters.pop(j)
    
    def _merge(self, indices):
        """
        Internal function to merge the biclusters inside of self.biclusters at specified indices
        
        :param indices: numpy.ndarray,
        Indices to merge together to form one bicluster
        
        :return: tuple of two numpy.ndarray masks representing the bicluster composed of merged biclusters
        """
        
        cols_ids_bic = np.full(self.biclusters[indices[0]][1].shape[0], False)
        rows_ids_bic = np.full(self.biclusters[indices[0]][0].shape[0], False)
        for i in indices:
            cols_ids_bic += self.biclusters[i][1]
            rows_ids_bic += self.biclusters[i][0]
        return rows_ids_bic, cols_ids_bic
    
    def _auto_select_nb_clusters(self):
        """
        Internal function to select the number of clusters when it's not provided
        """
        linkage_matrix = self._construct_linkage()
        s = linkage_matrix.shape
        height = linkage_matrix[:,2]
        shifted_height = np.array([0] + list(height[:s[0] - 1]))
        res = height - shifted_height
        best_ind = np.argmax(res)
        self.n_clusters = len(self.biclusters) - best_ind
        
    
    
    def _tree_select(self, indices):
        """
        Internal function to return the biclusters with the best scores for a selected branch
        """
        best_ind = np.argmax(self.score[indices])
        return self.biclusters[indices[best_ind]]
        
    def _construct_linkage(self):
        """
        Internal function to construct the linkage tree of biclusters 
        """
        dist_matrix = distance.distance_matrix(self.biclusters)
        dist_matrix = squareform(dist_matrix)
        linkage_matrix = linkage(dist_matrix , "ward")
        return linkage_matrix
    
    def tree_select_reduction(self):
        linkage_matrix = self._construct_linkage()
        assigned_clusters = cut_tree(linkage_matrix, n_clusters = self.n_clusters)
        cluster_values = np.unique(assigned_clusters)
        biclusters = []
        for i in cluster_values:
            ind = np.where(assigned_clusters == cluster_values[i])[0]
            b = self._tree_select(ind)
            biclusters.append(b)
        self.biclusters = biclusters
    
    def merge_reduction(self):
        """
        Merge the biclusters generated by the fit method in self.n_clusters clusters and update self.biclusters
        """
        linkage_matrix = self._construct_linkage()
        assigned_clusters = cut_tree(linkage_matrix, n_clusters = self.n_clusters)
        cluster_values = np.unique(assigned_clusters)
        biclusters = []
        for i in cluster_values:
            ind = np.where(assigned_clusters == cluster_values[i])[0]
            biclusters.append(self._merge(ind))
        self.biclusters = biclusters
        self.score = quality.score_biclusters(self.biclusters, self.data, self.var_type)
        
    def select_reduction(self):
        """
        Select the self.n_clusters clusters biclusters generated by the fit method and update self.biclusters
        """
        indices = np.argpartition(self.score, -self.n_clusters)[-self.n_clusters:]
        # Sorting indices according to the best quality biclusters
        indices = [ind for _, ind in sorted(zip(self.score, indices))]
        bic = self.biclusters
        self.biclusters = []
        for i in indices:
            self.biclusters.append(bic[i])
        self.score = self.score[indices]
        
    def reduce(self):
        """
        Reduce the number of biclusters found by the algorithm with either merging strategies or selection strategies
        
        """
        if len(self.biclusters) <= 1:
            return
        elif self.n_clusters == "auto":
            self._auto_select_nb_clusters()
            
            
        if self.reduction == "merge" and self.n_clusters < len(self.biclusters):
            self.merge_reduction()
        elif self.reduction == "selection" and self.n_clusters < len(self.biclusters):
            self.select_reduction()
        elif self.reduction == "tree_selection" and self.n_clusters < len(self.biclusters) :
            self.tree_select_reduction()

            
        
    def fit(self, data, var_type="Numeric"):
        """
        Use the Hbic algorithm to create biclusters


        :param data: numpy.ndarray containing all the data
        :param var_type: list of int indicating the type of each column of data

        """
        if type(var_type) == str:
            var_type = np.full(data.shape[1], var_type)
        var_type = np.array(var_type)
        self.var_type = var_type
        self.data = data
        n_rows, n_cols = data.shape
        self.n_mincols = max(int(n_cols * self.min_cols_prop), self.min_cols_abs)
        self.n_minrows = max(int(n_rows * self.min_rows_prop), self.min_rows_abs)
        self.biclusters = []

        arr_discretized = discretization.discretize(data, self.nbins, var_type)

        # We consider each column and each value of each column as a starting point
        for col_index in tqdm(range(n_cols), disable=not self.verbose):
            # Used in _find_best_column for optimisation purposes
            self.check = True

            col_values = arr_discretized[:, col_index]
            unique_col_values = set(col_values)
            for value in unique_col_values:
                # Rows init
                rows_ids_bic = col_values == value
                previous_size = 0
                size = sum(rows_ids_bic)
                if size < self.n_minrows:
                    continue

                # Cols init
                cols_ids_bic = np.full(n_cols, False)
                cols_ids_bic[col_index] = True
                biggest_bic = (rows_ids_bic.copy(), cols_ids_bic.copy())
                remaining_columns = np.where(cols_ids_bic == False)[0]
                while sum(remaining_columns) > 1:
                    temp_data = arr_discretized[rows_ids_bic, :]
                    val, column = self._find_best_column(temp_data, remaining_columns)
                    rows = arr_discretized[:, column] == val
                    self._add_column(cols_ids_bic, rows_ids_bic, column, rows)
                    # If even the best column does not share enough with the bicluster we can stop
                    if sum(rows_ids_bic) < self.n_minrows:
                        break
                    remaining_columns = np.delete(
                        remaining_columns, np.argmax(remaining_columns == column)
                    )

                    # Compare the size with the previous size and stops if it get smaller
                    size = sum(cols_ids_bic) * sum(rows_ids_bic)
                    if size <= previous_size:
                        break
                        
                    # Keep the biggest bic in memory
                    biggest_bic = (rows_ids_bic.copy(), cols_ids_bic.copy())
                    previous_size = size

                if previous_size > 0 and sum(biggest_bic[1]) > self.n_mincols:
                    self.biclusters.append(biggest_bic)
        self._remove_repeated_bic()
        
        if len(self.biclusters) > 1:
            self.score = quality.score_biclusters(self.biclusters, data, var_type)
        else:
            self.score = []
            
        self.reduce()
        
        
        
    def fit_predict(self, data, var_type="Numeric"):
        """
        Use the Hbic algorithm and returns biclusters

        :param data: numpy.array containing all the data
        :param var_type: list of int indicating the type of each column of data

        :return: list containing pairs of 1D numpy.array for each bicluster,  the
        first array is a mask on the rows and the second is a mask on the columns
        """

        self.fit(data, var_type)

        return self.biclusters

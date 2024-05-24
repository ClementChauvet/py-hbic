import numpy as np
from tqdm import tqdm
from collections.abc import Iterable

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
        random_state=None,
        column_proportion=1,
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
        self.column_proportion = column_proportion
        self.random_state = np.random.RandomState(random_state)

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

        # store a (length of columns,  4) matrix where first column is the count of the most frequent value in the column, second column is the value, 
        #third column is a boolean if the column is selected and fourth column is the index of the column in the original data
        
        if self.check:
            iteration_counts = np.zeros(unclustered_columns.shape)
            iteration_values = np.zeros(unclustered_columns.shape)
            selected_columns = np.zeros(unclustered_columns.shape)
            for i in range(len(unclustered_columns)):
                values, counts = np.unique(
                    arr_discretized[:, unclustered_columns[i]], return_counts=True
                )
                max_index = np.argmax(counts)
                iteration_counts[i] = counts[max_index]
                iteration_values[i] = values[max_index]
            self.iteration_matrix = np.column_stack((iteration_counts, iteration_values, selected_columns, unclustered_columns))
            self.iteration_matrix = self.iteration_matrix[self.iteration_matrix[:, 0].argsort()]
            self.iteration_matrix = self.iteration_matrix[::-1]
            value, column = self.iteration_matrix[0,1], int(self.iteration_matrix[0,3])
            self.iteration_matrix = self.iteration_matrix[1:]
            self.check = False
            return value, column


        if self.iteration_matrix.shape[0] == 1:
            values, counts = np.unique(
                    arr_discretized[:, int(self.iteration_matrix[0, 3])], return_counts=True
                )
            max_index = np.argmax(counts)
            value = values[max_index]
            column = int(self.iteration_matrix[0, 3])
            return value, column
        
        if self.iteration_matrix.shape[0] == 0:
            raise ValueError("No column found")
        

        current_comparison = 1
        while True:
            values, counts = np.unique(
                    arr_discretized[:, int(self.iteration_matrix[0, 3])], return_counts=True
                )
            max_index = np.argmax(counts)
            if self.iteration_matrix[current_comparison, 0] <= counts[max_index]:
                value = values[max_index]
                column = int(self.iteration_matrix[0, 3])
                self.iteration_matrix = self.iteration_matrix[1:]
                return value,column
            else:
                self.iteration_matrix[0, 0] = counts[max_index]
                self.iteration_matrix[0, 1] = values[max_index]
                self.iteration_matrix = self.iteration_matrix[self.iteration_matrix[:, 0].argsort()]
                self.iteration_matrix = self.iteration_matrix[::-1]

        raise ValueError("No column found")

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


        
    def _select_pareto_front(self):
        """
        Internal function to select biclusters in the pareto front and discard others
        """
        quality_scores = quality.quality_evaluation_biclusters(self.biclusters, self.data, self.var_type)
        size_scores = quality.sizes(self.biclusters)
        #normalisation 
        quality_scores = (quality_scores / max(max(quality_scores), 1))
        size_scores = 1 - (size_scores / max(max(size_scores), 1))
        scores = zip(quality_scores, size_scores)
        pareto_optimal_ind = distance.is_pareto_efficient(np.array(list(scores)))
        self.biclusters = [self.biclusters[i] for i in range(len(self.biclusters)) if pareto_optimal_ind[i]]

    def _select_distance(self, n_bic):
        """
        Internal function to select biclusters by distance L2 distance to the origin of a quality, size graph
        The selected biclusters are the n bics that are the closest to the origin, with n being the biggest gap in quality
        """
        scores = quality.L2_score_biclusters(self.biclusters, self.data, self.var_type)
        sorted_scores = np.sort(scores)[::-1]
        if n_bic is None:
            if sorted_scores[0] == np.mean(sorted_scores): #If score is constant we keep all biclusters
                n_bic = len(sorted_scores)
            else:
                differences = np.diff(sorted_scores)[::-1]
                n_bic = len(differences) - np.argmin(differences) 
        selected = [bic for _, bic in sorted(zip(scores, self.biclusters), key = lambda t: t[0])][-n_bic:]
        self.biclusters = [s for s in selected]

        
    def reduce(self, n_clusters):
        """
        Reduce the number of biclusters found to self.n_clusters

        """
        if len(self.biclusters) == 0:
            return
        if self.reduction == "pareto":
            self._select_pareto_front()
        elif self.reduction == "distance":
            self._select_distance(n_clusters)
        elif self.reduction == None:
            return
        else:
            raise ValueError(
                "reduction parameter must be 'distance', 'pareto' or None"
            )

            
        
    def fit(self, data, var_type=None, n_clusters = None):
        """
        Use the Hbic algorithm to create biclusters


        :param data: numpy.ndarray containing all the data
        :param var_type: list of int indicating the type of each column of data

        """
        if var_type is None:
            var_type = discretization.infer_var_type(data)
        elif type(var_type) == str:
            var_type = np.full(data.shape[1], var_type)
        var_type = np.array(var_type)
        self.var_type = var_type
        self.data = data
        n_rows, n_cols = data.shape
        self.n_mincols = max(int(n_cols * self.min_cols_prop), self.min_cols_abs)
        self.n_minrows = max(int(n_rows * self.min_rows_prop), self.min_rows_abs)
        self.biclusters = []

        
        arr_discretized = discretization.discretize(data, self.nbins, var_type)
        starting_columns = range(n_cols)
        starting_columns = self.random_state.choice(starting_columns, int(n_cols * self.column_proportion), replace=False)

        # We consider each column and each value of each column as a starting point
        for col_index in tqdm(starting_columns, disable=not self.verbose):
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
        
        self.reduce(n_clusters)
        
        
    def fit_predict(self, data, var_type=None, n_clusters = None):
        """
        Use the Hbic algorithm and returns biclusters

        :param data: numpy.array containing all the data
        :param var_type: list of int indicating the type of each column of data

        :return: list containing pairs of 1D numpy.array for each bicluster,  the
        first array is a mask on the rows and the second is a mask on the columns
        """

        self.fit(data, var_type)

        return self.biclusters

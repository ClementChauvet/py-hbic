import os
from tqdm import tqdm
import pandas as pd
import copy
import numpy as np

from hbic import Hbic
import hbic.utils.metrics as metrics

try:
    os.chdir(os.path.dirname(__file__))
except NameError:
    pass
    


def aggregate(results):
    try:
        os.mkdir("results")
    except:
        pass
    
    dirs = results["dir"].unique()
    columns = ["sub_dir", "reduce", "cluster_detection" ,"beta", "rec", "rel", "ayadi", "l_and_w"]
    for i in range(len(dirs)):
        df_dir = results[results["dir"] == dirs[i]]
        reduces = df_dir["reduce"].unique()
        sub_dfs = []
        for k in range(len(reduces)): 
            df_reduce = df_dir[df_dir["reduce"] == reduces[k]]
            cluster_detections = df_reduce["cluster_detection"].unique()
            for l in range(len(cluster_detections)):
                res_df = pd.DataFrame(columns = columns)
                df_cluster_reduction = df_reduce[df_reduce["cluster_detection"] == cluster_detections[l]]
                sub_dirs = df_cluster_reduction["sub_dir"].unique()
                for j in range(len(sub_dirs)):       
                    df_sub_dir = df_cluster_reduction[df_cluster_reduction["sub_dir"] == sub_dirs[j]]
                    res_df.loc[len(res_df)] = df_sub_dir.mean(numeric_only = True)
                    res_df.loc[len(res_df)-1, "sub_dir"] = sub_dirs[j]
                    res_df.loc[len(res_df)-1, "cluster_detection"] = cluster_detections[l]
                    res_df.loc[len(res_df)-1, "reduce"] = reduces[k]
                res_df.columns = pd.MultiIndex.from_product([[reduces[k]], res_df.columns])
                sub_dfs.append(res_df)
        df = pd.concat(sub_dfs, axis = 1)
        df.to_excel("results/" + dirs[i] + "_results_benchmark.xlsx")
        
             


def preprocess_num(data_file):
    df = pd.read_csv(data_file, sep = '\t')
    for i in df.columns:
        if pd.api.types.is_string_dtype(df[i]):
            df[i] = df[i].astype("category").cat.codes

    arr = df.to_numpy()
    return arr
    

def get_ref_biclusters(nbics, shape, d = "het-data", size = 75):
    ref_bic = []
    decal = 0
    if d == "het-data":
        row_size = 50
        col_size = 30
    if d == "size":
        decal = 1
        row_size = size
        col_size = size
    if d == "number":
        row_size = 50
        col_size = 50
    if d == "overlap":
        row_size = 50
        col_size = 50
    if d == "noise":
        row_size = 50
        col_size = 50
    for i in range(nbics):
        rows, cols = np.full(shape[0], False), np.full(shape[1], False)
        rows[i * row_size:(i+1) * row_size] = True
        cols[decal + i * col_size:decal + (i+1) * col_size] = True
        ref_bic.append((rows, cols))
    return ref_bic


def append_results(df, bic, ref_bic, cluster_detection, reduce, directory, sub_dir, example):
    dic = {}
    dic["cluster_detection"] = str(cluster_detection)
    dic["dir"] = directory
    dic["sub_dir"] = sub_dir
    dic["example"] = example
    dic["reduce"] = reduce
    dic["rel"] = metrics.prelic_relevance(bic, ref_bic)
    dic["beta"] = len(bic)
    dic["rec"] = metrics.prelic_recovery(bic, ref_bic)
    dic["l_and_w"] = metrics.l_and_w(bic,ref_bic)
    dic["ayadi"] = metrics.ayadi(bic, ref_bic)
    df.loc[len(df)] = dic
    return df

  
if __name__ == '__main__':
    n_clusters = {
        "size" : [1,1],
         "number": [1,2,3,4,5],
        "overlap" : [2,2,2, 2 ],
        "noise" : [1, 1, 1, 1]
    }
    
    n_bins = {
        "size" : [15,10],
         "number": [10,10,10,10,10],
        "overlap" : [12,12,12, 12],
        "noise" : [4, 4, 4, 4]
        
    }
    agg = ["all", "merge", "selection", "tree_selection"]
    root = "datasets/padilha_constant/"
    dirs = ["size", "number", "overlap", "noise"]
    results = pd.DataFrame(columns = ["dir","sub_dir", "example", "reduce", "cluster_detection" ,"beta", "rec", "rel", "ayadi", "l_and_w"])
    for dir in dirs:
        print(dir)
        sub_dirs = os.listdir(root + dir)
        for i in range(len(sub_dirs)):
            path = root + dir + "/"+ sub_dirs[i] +"/"
            examples = os.listdir(path)
            arr = preprocess_num(path + examples[0])
            if dir == "size":
                ref_bic = get_ref_biclusters(n_clusters[dir][i],arr.shape, dir, size = int(sub_dirs[i][:2]))
            else :
                ref_bic = get_ref_biclusters(n_clusters[dir][i],arr.shape, dir)

            res = Hbic(verbose = False, reduction = None, n_clusters = n_clusters[dir][i], nbins = n_bins[dir][i])
            progress_bar = tqdm(range(len(examples)))
            progress_bar.set_description("Processing " + sub_dirs[i])
            for j in progress_bar:
                arr = preprocess_num(path + examples[j])
                bic = res.fit_predict(arr)
                for agg_i in range(len(agg)):
                    for cluster_detection in [True, False]:
                        res2 = copy.deepcopy(res)
                        res2.reduction = agg[agg_i]
                        res2.n_clusters = "auto" if cluster_detection else n_clusters[dir][i]
                        res2.reduce()
                        bic = res2.biclusters
                        results = append_results(results, bic, ref_bic, cluster_detection, agg[agg_i], dir, sub_dirs[i], examples[j])
        
        aggregate(results)
import numpy as np

def prelic(bic_found,bic_ref, axis = 0):
    r = 0
    for bic in bic_found:
        rbest = 0
        for ref in bic_ref:
            union, inter = bic[axis] + ref[axis], bic[axis] * ref[axis]
            union, inter = np.where(union)[0], np.where(inter)[0]
            rbest = max(rbest, inter.shape[0]/union.shape[0])
        r+= rbest
    return r / max(len(bic_found),1)

def prelic_relevance(bic_found, bic_ref):
    p = prelic(bic_found,bic_ref, axis = 0) * prelic(bic_found,bic_ref, axis = 1)
    p = np.sqrt(p)
    return p

def prelic_recovery(bic_found, bic_ref):
    return prelic_relevance(bic_ref, bic_found)


def ayadi(bic_found,bic_ref):
    r = 0
    for bic in bic_found:
        rbest = 0
        for ref in bic_ref:
            inter_row, inter_col = bic[0] * ref[0], bic[1] * ref[1]
            union_row, union_col = bic[0] + ref[0], bic[1] + ref[1]
            inter_row, inter_col, union_row, union_col = np.where(inter_row)[0], np.where(inter_col)[0],np.where(union_row)[0],np.where(union_col)[0]
            rbest = max(rbest, (inter_row.shape[0] * inter_col.shape[0]) /  (union_row.shape[0] * union_col.shape[0]))
        r+= rbest
    return  r / max(len(bic_found),1)

def l_and_w(bic_found,bic_ref):
    r = 0
    for bic in bic_found:
        rbest = 0
        for ref in bic_ref:
            inter_row, inter_col = bic[0] * ref[0], bic[1] * ref[1]
            union_row, union_col = bic[0] + ref[0], bic[1] + ref[1]
            inter_row, inter_col, union_row, union_col = np.where(inter_row)[0], np.where(inter_col)[0],np.where(union_row)[0],np.where(union_col)[0]            
            rbest = max(rbest, (inter_row.shape[0] + inter_col.shape[0]) /  (union_row.shape[0] + union_col.shape[0]))
        r+= rbest
    return r / max(len(bic_found),1)

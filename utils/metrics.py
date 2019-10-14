import numpy as np

def calc_confusion_matrix(mask, pred):
    combined = mask * 2 + pred

    fp_total = 0 # false positive total pixels
    fn_total = 0 # false negative total pixels
    tn_total = 0 # true negative total pixels
    tp_total = 0 # true positive total pixels

    for image in combined:
        fp = np.count_nonzero(image == 1.0) # false positive (red)
        fn = np.count_nonzero(image == 2.0) # false negative (yellow)
        tp = np.count_nonzero(image == 3.0) # true positive (green)
        tn = np.count_nonzero(image == 0.0) # true negative (black)

        fp_total += fp
        fn_total += fn
        tn_total += tn
        tp_total += tp

    fpr_perc = fp_total / (fp_total + tn_total) if tn_total != 0 or fp_total != 0 else 0
    fnr_perc = fn_total / (fn_total + tp_total) if tp_total != 0 or fn_total != 0 else 0

    f_total = fp_total + fn_total

    fpr_perc = '{:.2%}'.format(fpr_perc)
    fnr_perc = '{:.2%}'.format(fnr_perc)

    conf = dict(
        fpr_perc = fpr_perc,
        fnr_perc = fnr_perc,
        fp_total = fp_total,
        fn_total = fn_total,
        tp_total = tp_total,
        tn_total = tn_total,
        f_total = f_total
    )

    return conf

def calc_fp_rate(fp_total, tn_total):
    fp_rate = fp_total / (fp_total + tn_total) if tn_total != 0 or fp_total != 0 else 0

    return fp_rate

def calc_fn_rate(fn_total, tp_total):
    fn_rate = fn_total / (fn_total + tp_total) if tp_total != 0 or fn_total != 0 else 0

    return fn_rate
# -*-coding:utf8-*-

def evaluate_results_binary(result, neg_id):
    total_p, total_g, right, total, total_right = 0, 0, 0, 0, 0
    for _p, g in result:
        total += len(_p)
        if g[0] != neg_id:
            total_g += len(g)
        for p in _p:
            if p != neg_id: total_p += 1
            if p in g: total_right += 1
            if p != neg_id and p in g: right += 1
    if total_p == 0: total_p = 1
    acc = 1.0 * total_right / total
    pre = 1.0 * right / total_p
    rec = 1.0 * right / total_g
    f1 = 2 * pre * rec / (pre + rec + 0.000001)
    out = 'Total Sample:%d, Total Pred: %d, Total Right_pred:%d, Right Pred_evt: %d, Total Event: %d\n' % (
    total, total_p, total_right, right, total_g)
    out += 'Accurate:%.3f, Precision: %.3f, Recall: %.3f, F1: %.3f' % (acc, pre, rec, f1)
    return out, f1

from attrdict import AttrDict

def compute_metrics(preds, trues):
    '''Computes main metrics for the prediction.
    
    Args:
        preds (numpy.array): predictions array, with shape (N,)
        trues (numpy.array): ground truth array, with shape (N,)
    
    Returns:
        dict: dictionary with metrics
    '''
    
    tp = ((trues == 1) * (preds == 1)).sum()
    tn = ((trues == 0) * (preds == 0)).sum()
    fp = ((trues == 0) * (preds == 1)).sum()
    fn = ((trues == 1) * (preds == 0)).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    return AttrDict({
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'specificity': round(specificity, 2),       
    })
    

def specificity_score(preds, trues):
    '''Computes specificity score for the prediction. Need this because sklearn's 'cross_val_score'
    and 'cross_validate' don't have this metric.'''
    
    tp = ((trues == 1) * (preds == 1)).sum()
    tn = ((trues == 0) * (preds == 0)).sum()
    fp = ((trues == 0) * (preds == 1)).sum()
    fn = ((trues == 1) * (preds == 0)).sum()
    
    specificity = tn / (tn + fp)
    
    return specificity
    
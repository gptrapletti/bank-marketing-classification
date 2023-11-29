import random
from imblearn.over_sampling import SMOTE

def downsampling(X, y, factor, seed=42):
    """Gets a dataset and returns a downsampled version of it. For unbalanced datasets where 0 is 
    the majority class and 1 the minority class.
    
    Args:
        X (pandas.DataFrame): pandas dataframe with features.
        y (pandas.Series): pandas series with labels.
        factor (float): downsampling factor. Multiplier for the number of majority class samples 
            relative to the minority class count. E.g., with `factor=2`, class 0 will have
            twice the number of instances then class 1.
        seed: random seed for reproducibility.
    
    Returns:
        X_downsampled (pandas.DataFrame): downsampled features.
        y_downsampled (pandas.Series): downsampled labels.    
    """
    idxs_target_1 = y[y == 1].index.tolist()
    idxs_target_0 = y[y == 0].index.tolist()
    
    n_downsampling = int(len(idxs_target_1) * factor)
    
    random.seed(seed)
    idxs_target_0_to_keep = random.sample(idxs_target_0, n_downsampling)
    
    idxs = list(set(idxs_target_1) | set(idxs_target_0_to_keep))
    X_downsampled = X.loc[idxs].reset_index(drop=True)
    y_downsampled = y.loc[idxs]
    
    return X_downsampled, y_downsampled


def upsampling(X, y, factor, seed=42):
    """Gets a dataset and returns an upsampled version of it. For unbalanced datasets where 0 is 
    the majority class and 1 the minority class.
    First, class 1 is upsampled by a factor of 'factor', then class 0 is downsampled to match the
    number of instances of class 1, thus producing a perfectly balanced dataset.
    
    Args:
        X (pandas.DataFrame): pandas dataframe with features.
        y (pandas.Series): pandas series with labels.
        factor (float): upsampling factor for class 1. If factor=1, class 1 instances in the 
            final resampled dataset are doubled; if factor=2, they are tripled. 
        seed: random seed for reproducibility.
    
    Returns:
        X_upsampled (pandas.DataFrame): upsampled features.
        y_upsampled (pandas.Series): upsampled labels.    
    """
    idxs_target_1 = y[y == 1].index.tolist()
    idxs_target_0 = y[y == 0].index.tolist()

    # Upsample class 1
    n_to_upsample_target_1 = int(len(idxs_target_1) * factor)
    random.seed(seed)
    upsampled_idxs_target_1 = random.choices(population=idxs_target_1, k=n_to_upsample_target_1)
    final_idxs_target_1 = idxs_target_1 + upsampled_idxs_target_1

    # Downsample class 0
    random.seed(seed)
    idxs_target_0_to_keep = random.sample(idxs_target_0, len(final_idxs_target_1))

    idxs = final_idxs_target_1 + idxs_target_0_to_keep
    X_upsampled = X.loc[idxs].reset_index(drop=True)
    y_upsampled = y.loc[idxs]

    return X_upsampled, y_upsampled


def smote_sampling(X, y, factor, neighbors=5, seed=42):
    """Gets a dataset and returns a smote-sampled version of it. For unbalanced datasets where 0 is 
    the majority class and 1 the minority class.
    
    SMOTE will balance the dataset by creating synthetic data with class 1 by KNN. To avoid a too 
    high proportion of synthetic data, class 0 is first downsampled to N, where N is the number of 
    samples per class I'll want in the final dataset. This is here expressed as a multiple of the 
    number of instances of class 1 in the original dataset, controlled by the 'factor' argument. 
    
    First, the 'factor' argument is a multiplier used to define the number of class 1 instances the
    final resampled dataset will have (N). Then, class 0 instances are downsampled to match this 
    number (N). Lastly, SMOTE is applied to create as many synthetic class 1 instances as necessary
    to reach N, thus producing a perfectly balanced dataset (N class 1 instances and N class 1
    instances).
    
    Args:
        X (pandas.DataFrame): pandas dataframe with features.
        y (pandas.Series): pandas series with labels.
        factor (float): sampling factor for class 1. If 'factor=2', the final resampled dataset will
            have twice the number of class 1 instances than the original dataset (and as many class
            0 instances).
    
    Returns:
        X_upsampled (pandas.DataFrame): upsampled features.
        y_upsampled (pandas.Series): upsampled labels.    
    """
    idxs_target_1 = y[y == 1].index.tolist()
    idxs_target_0 = y[y == 0].index.tolist()

    # Downsample class 0
    n_downsampling = len(idxs_target_1) * factor
    random.seed(seed)
    idxs_target_0_to_keep = random.sample(idxs_target_0, n_downsampling)
    idxs = list(set(idxs_target_1) | set(idxs_target_0_to_keep))
    X_downsampled = X.loc[idxs].reset_index(drop=True)
    y_downsampled = y.loc[idxs]

    smote = SMOTE(
        sampling_strategy='minority', # resample only the minority class
        k_neighbors=neighbors, 
        random_state=seed
    )

    X_resampled, y_resampled = smote.fit_resample(X_downsampled, y_downsampled)

    return X_resampled, y_resampled

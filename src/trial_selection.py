import numpy as np

def get_signed_contrast(trials):
    cl = np.nan_to_num(trials['contrastLeft'])
    cr = np.nan_to_num(trials['contrastRight'])
    return cl-cr

def make_condition_masks(signed_contrast, min_trials=5):
    levels = np.sort(np.unique(signed_contrast))
    masks = {}
    for c in levels:
        mask = np.isclose(signed_contrast, c)
        if np.sum(mask) >= min_trials:
            masks[c] = mask
    return masks
 
def get_high_masks(signed_contrast, min_trials=5, threshold=0.5):
    high_mask = np.abs(signed_contrast) >= threshold
    if np.sum(high_mask) < min_trials:
        print(f'Warning: only{np.sum(high_mask)} trials above threshold {threshold}')
        return None
    else: 
        return high_mask

def get_null_masks(trials, signed_contrast, min_trials=5):
    cl = np.nan_to_num(trials['contrastLeft'])
    cr = np.nan_to_num(trials['contrastRight'])
    null_mask = (cl == 0) & (cr == 0)
    if np.sum(null_mask) < min_trials:
        print(f'Warning: only {np.sum(null_mask)} null trials')
        return None
    else:
        return null_mask

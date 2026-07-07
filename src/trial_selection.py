# trial_selection.py
import numpy as np

def get_signed_contrast(trials):
    cl = np.nan_to_num(trials['contrastLeft'], nan=0.0)
    cr = np.nan_to_num(trials['contrastRight'], nan=0.0)
    return cl-cr

def make_condition_masks(signed_contrast, min_trials=5):
    levels = np.sort(np.unique(signed_contrast))
    masks = {}
    for c in levels:
        mask = np.isclose(signed_contrast, c)
        if np.sum(mask) >= min_trials:
            masks[c] = mask
    return masks

def make_contrast_pair_condition_masks(trials, min_trials=5):
    contrast_left = np.nan_to_num(trials['contrastLeft'])
    contrast_right = np.nan_to_num(trials['contrastRight'])
    cl = np.asarray(contrast_left, float)
    cr = np.asarray(contrast_right, float)

    conditions = sorted(set(zip(cl, cr)))
    masks = {}
    for cond in conditions:
        c_left, c_right = cond
        mask = np.isclose(cl, c_left) & np.isclose(cr, c_right)
        if np.sum(mask) >= min_trials:
            masks[cond] = mask
    valid_conditions = list(masks.keys())
    return valid_conditions, masks

def get_high_masks(signed_contrast, min_trials=5, threshold=0.5):
    high_mask = np.abs(signed_contrast) >= threshold
    if np.sum(high_mask) < min_trials:
        print(f'Warning: only{np.sum(high_mask)} trials above threshold {threshold}')
        return None
    else: 
        return high_mask
    
def get_pos_neg_masks(signed_contrast, high_mask=None, min_trials=5):
    if high_mask is not None:
        pos_mask = high_mask & (signed_contrast > 0)
        if np.sum(pos_mask) < min_trials:
            raise ValueError(f'Warning: only {np.sum(pos_mask)} positive trials')
        neg_mask = high_mask & (signed_contrast < 0)
        if np.sum(neg_mask) < min_trials:
            raise ValueError(f'Warning: only {np.sum(neg_mask)} negative trials')
    else:
        pos_mask = signed_contrast > 0
        if np.sum(pos_mask) < min_trials:
            raise ValueError(f'Warning: only {np.sum(pos_mask)} positive trials')
        neg_mask = signed_contrast < 0
        if np.sum(neg_mask) < min_trials:
            raise ValueError(f'Warning: only {np.sum(neg_mask)} negative trials')
    return pos_mask, neg_mask

def get_null_masks(trials, signed_contrast, min_trials=5):
    cl = np.nan_to_num(trials['contrastLeft'])
    cr = np.nan_to_num(trials['contrastRight'])
    null_mask = (cl == 0) & (cr == 0)
    if np.sum(null_mask) < min_trials:
        print(f'Warning: only {np.sum(null_mask)} null trials')
        return None
    else:
        return null_mask


import numpy as np
from sklearn.metrics import roc_auc_score


def image_level_auroc(labels, scores):
    labels = np.array(labels)
    scores = np.array(scores)
    if len(np.unique(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores)


def pixel_level_auroc(gt_masks, pixel_maps):
    
    all_gt, all_pred = [], []
    for mask, pmap in zip(gt_masks, pixel_maps):
        all_gt.append(np.asarray(mask, dtype=np.uint8).ravel())
        all_pred.append(np.asarray(pmap).ravel())

    all_gt   = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    if len(np.unique(all_gt)) < 2:
        return float('nan')
    return roc_auc_score(all_gt, all_pred)


def summarize_results(results_dict):
    """
    results_dict: {
      category: {
        model: {
          'clean':       {'img_auroc': float, 'pix_auroc': float},
          'corruption_name_sev': {'img_auroc': float, 'pix_auroc': float},
          ...
        }
      }
    }
    Returns a flat list of rows suitable for pandas / tabulate.
    """
    rows = []
    for category, models in results_dict.items():
        for model_name, conditions in models.items():
            for condition, metrics in conditions.items():
                rows.append({
                    'category':   category,
                    'model':      model_name,
                    'condition':  condition,
                    'img_auroc':  round(metrics['img_auroc'], 4),
                    'pix_auroc':  round(metrics.get('pix_auroc', float('nan')), 4),
                })
    return rows

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Convert normalised CHW tensor -> HWC uint8 numpy array."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def save_heatmap_grid(images, pixel_maps, labels, gt_masks, title, save_path,
                      n_cols=4, n_rows=2):
    """
    Save a grid showing: input | anomaly heatmap | ground-truth mask.
    Picks up to n_cols*n_rows images (anomalous preferred).
    """
    anomaly_idx = [i for i, l in enumerate(labels) if l == 1]
    normal_idx  = [i for i, l in enumerate(labels) if l == 0]
    # Take a mix, anomalous first
    chosen = (anomaly_idx + normal_idx)[:n_cols * n_rows]
    if len(chosen) == 0:
        return

    n_show = len(chosen)
    n_cols_actual = min(n_cols, n_show)
    n_rows_actual = -(-n_show // n_cols_actual)  # ceil division

    fig, axes = plt.subplots(n_rows_actual, n_cols_actual * 3,
                             figsize=(n_cols_actual * 7, n_rows_actual * 2.5))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for pos, idx in enumerate(chosen):
        row = pos // n_cols_actual
        col = pos % n_cols_actual

        img = images[idx]
        pmap = pixel_maps[idx]
        mask = gt_masks[idx]
        lbl  = labels[idx]

        ax_img  = axes[row, col * 3]
        ax_heat = axes[row, col * 3 + 1]
        ax_mask = axes[row, col * 3 + 2]

        ax_img.imshow(img)
        ax_img.set_title('Input\n(' + ('Defect' if lbl else 'Good') + ')', fontsize=7)
        ax_img.axis('off')

        ax_heat.imshow(img)
        ax_heat.imshow(pmap, cmap='jet', alpha=0.5)
        ax_heat.set_title('Anomaly Map', fontsize=7)
        ax_heat.axis('off')

        if mask is not None:
            ax_mask.imshow(mask, cmap='gray')
            ax_mask.set_title('GT Mask', fontsize=7)
        else:
            ax_mask.imshow(np.zeros_like(pmap), cmap='gray')
            ax_mask.set_title('GT Mask\n(none)', fontsize=7)
        ax_mask.axis('off')

    # Hide unused axes
    for pos in range(n_show, n_rows_actual * n_cols_actual):
        row = pos // n_cols_actual
        col = pos % n_cols_actual
        for k in range(3):
            axes[row, col * 3 + k].axis('off')

    fig.suptitle(title, fontsize=10, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_auroc_vs_severity(results, corruption_names, model_names,
                           metric='img_auroc', save_path=None):
    
    from .corruptions import N_SEVERITIES

    ylabel = 'Image-level ROC-AUC' if metric == 'img_auroc' else 'Pixel-level ROC-AUC'

    for model in model_names:
        # --- clean AUC (averaged over categories) ---
        clean_aurocs = []
        for cat_data in results.values():
            if model in cat_data and 'clean' in cat_data[model]:
                v = cat_data[model]['clean'].get(metric, np.nan)
                if not np.isnan(v):
                    clean_aurocs.append(v)
        clean_auc = np.nanmean(clean_aurocs) if clean_aurocs else np.nan

        fig, ax = plt.subplots(figsize=(7, 5))

        for corr in corruption_names:
            vals = []
            for sev in range(1, N_SEVERITIES + 1):
                key = f'{corr}_sev{sev}'
                aurocs = []
                for cat_data in results.values():
                    if model in cat_data and key in cat_data[model]:
                        v = cat_data[model][key].get(metric, np.nan)
                        if not np.isnan(v):
                            aurocs.append(v)
                vals.append(np.nanmean(aurocs) if aurocs else np.nan)
            ax.plot(range(1, N_SEVERITIES + 1), vals, marker='o',
                    label=corr.replace('_', ' '))

        if not np.isnan(clean_auc):
            ax.axhline(clean_auc, linestyle='--', color='steelblue', linewidth=1.2,
                       label=f'Clean AUC = {clean_auc:.3f}')

        ax.set_xlabel('Severity')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Robustness Results - {model}')
        ax.set_xticks(range(1, N_SEVERITIES + 1))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            out = f'{base}_{model.lower().replace(" ", "_")}{ext}'
            os.makedirs(os.path.dirname(out), exist_ok=True)
            plt.savefig(out, dpi=130, bbox_inches='tight')
        plt.close(fig)


def plot_corruption_bar(results, categories, corruption_names, model_names,
                        metric='img_auroc', save_path=None):
    """
    Bar chart: AUROC drop per corruption type (clean − corrupted_sev3).
    Averaged over categories.
    """
    x = np.arange(len(corruption_names))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(model_names):
        drops = []
        for corr in corruption_names:
            clean_vals, corr_vals = [], []
            for cat, cat_data in results.items():
                if model not in cat_data:
                    continue
                if 'clean' in cat_data[model]:
                    clean_vals.append(cat_data[model]['clean'][metric])
                key = f'{corr}_sev3'
                if key in cat_data[model]:
                    corr_vals.append(cat_data[model][key][metric])
            if clean_vals and corr_vals:
                drop = np.nanmean(clean_vals) - np.nanmean(corr_vals)
            else:
                drop = 0.0
            drops.append(drop)
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, drops, width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in corruption_names], fontsize=9)
    ax.set_ylabel(f'AUROC Drop (clean − sev3)')
    ax.set_title(f'Performance Drop by Corruption Type\n(averaged over {categories})')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def plot_augmentation_comparison(results_baseline, results_augmented,
                                 categories, corruption_names,
                                 model_names, metric='img_auroc',
                                 save_path=None):
    
    from .corruptions import N_SEVERITIES

    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(5 * len(model_names), 4), sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, model in zip(axes, model_names):
        labels_c, base_vals, aug_vals = [], [], []
        for corr in corruption_names:
            bl_list, au_list = [], []
            for cat in categories:
                for sev in range(1, N_SEVERITIES + 1):
                    key = f'{corr}_sev{sev}'
                    if (cat in results_baseline and model in results_baseline[cat]
                            and key in results_baseline[cat][model]):
                        bl_list.append(results_baseline[cat][model][key][metric])
                    if (cat in results_augmented and model in results_augmented[cat]
                            and key in results_augmented[cat][model]):
                        au_list.append(results_augmented[cat][model][key][metric])
            labels_c.append(corr.replace('_', '\n'))
            base_vals.append(np.nanmean(bl_list) if bl_list else np.nan)
            aug_vals.append(np.nanmean(au_list) if au_list else np.nan)

        x = np.arange(len(labels_c))
        ax.bar(x - 0.2, base_vals, 0.35, label='Baseline', color='steelblue')
        ax.bar(x + 0.2, aug_vals,  0.35, label='+ Augmentation', color='darkorange')
        ax.set_xticks(x)
        ax.set_xticklabels(labels_c, fontsize=8)
        ax.set_title(model, fontsize=10)
        ax.set_ylim(0.4, 1.02)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(f'Avg {metric.upper()} (all severities)')
    plt.suptitle('Baseline vs Corruption-Aware Augmentation', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def save_roc_curves(labels, scores_dict, title, save_path):
    """labels: list[int], scores_dict: {name: list[float]}"""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

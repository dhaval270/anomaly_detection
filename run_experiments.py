import os, sys, json, csv, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'datasets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR     = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)

from src.dataset     import MVTecDataset, IMG_SIZE
from src.corruptions import (CORRUPTIONS, CORRUPTION_LABELS, N_SEVERITIES,
                              get_corruption_fn, apply_random_corruption)
from src.padim       import PaDiM
from src.patchcore   import PatchCore
from src.metrics     import image_level_auroc, pixel_level_auroc
from src.visualize   import (save_heatmap_grid, plot_auroc_vs_severity,
                              plot_corruption_bar, plot_augmentation_comparison,
                              save_roc_curves, denormalize)
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CATEGORIES  = ['bottle', 'capsule', 'carpet']
MODEL_NAMES = ['PaDiM', 'PatchCore']
BATCH_SIZE  = 8
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# -- Utilities -----------------------------------------------------------------

def make_loader(category, split, corruption_fn=None):
    ds = MVTecDataset(DATASET_DIR, category, split=split,
                      transform=TRANSFORM, corruption_fn=corruption_fn)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def build_model(name):
    if name == 'PaDiM':
        return PaDiM(device=DEVICE)
    return PatchCore(device=DEVICE)


def train_model(model, category, augment=False):
    fn = apply_random_corruption if augment else None
    loader = make_loader(category, 'train', corruption_fn=fn)
    tag = ' (augmented)' if augment else ''
    print(f'  Fitting {type(model).__name__} on {category}{tag}...')
    model.fit(loader)


def run_model_on(model, category, corruption_fn=None, tag='clean'):
    """Evaluate model, return metrics dict + raw results."""
    loader = make_loader(category, 'test', corruption_fn=corruption_fn)
    scores, pmaps, labels, masks = model.score(loader)
    img_auc = image_level_auroc(labels, scores)
    pix_auc = pixel_level_auroc(masks, pmaps)
    print(f'    [{tag:35s}]  img={img_auc:.4f}  pix={pix_auc:.4f}')
    return {
        'img_auroc': img_auc,
        'pix_auroc': pix_auc,
        'scores':    scores,
        'pmaps':     pmaps,
        'labels':    labels,
        'masks':     masks,
    }


def collect_raw_images(category, corruption_fn=None):
    """Return list of HWC uint8 numpy arrays for visualisation."""
    ds = MVTecDataset(DATASET_DIR, category, split='test',
                      transform=TRANSFORM, corruption_fn=corruption_fn)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    return [denormalize(b[0][0]) for b in loader]


# -- Phase 1 & 2 ---------------------------------------------------------------

def run_baseline_and_robustness():
    results = {c: {m: {} for m in MODEL_NAMES} for c in CATEGORIES}

    for cat in CATEGORIES:
        print(f'\n{"="*60}\nCategory: {cat}\n{"="*60}')
        raw_clean = collect_raw_images(cat)          # PIL numpy images (no corrupt)

        for mname in MODEL_NAMES:
            print(f'\n-- {mname} --')
            model = build_model(mname)
            train_model(model, cat, augment=False)

            # Clean
            r = run_model_on(model, cat, tag='clean')
            results[cat][mname]['clean'] = {'img_auroc': r['img_auroc'],
                                            'pix_auroc': r['pix_auroc']}
            save_heatmap_grid(
                raw_clean, r['pmaps'], r['labels'], r['masks'],
                title=f'{mname} - {cat} - Clean',
                save_path=os.path.join(FIG_DIR, f'{cat}_{mname}_clean_heatmap.png'))
            save_roc_curves(
                r['labels'], {mname: r['scores']},
                title=f'ROC - {mname} - {cat} - Clean',
                save_path=os.path.join(FIG_DIR, f'{cat}_{mname}_clean_roc.png'))

            # Corrupted
            for corr_name in CORRUPTIONS:
                for sev in range(1, N_SEVERITIES + 1):
                    tag = f'{corr_name}_sev{sev}'
                    fn  = get_corruption_fn(corr_name, sev)
                    rc  = run_model_on(model, cat, corruption_fn=fn, tag=tag)
                    results[cat][mname][tag] = {'img_auroc': rc['img_auroc'],
                                                'pix_auroc': rc['pix_auroc']}
                    # Save heatmap only at severity 3
                    if sev == 3:
                        raw_c = collect_raw_images(cat, corruption_fn=fn)
                        save_heatmap_grid(
                            raw_c, rc['pmaps'], rc['labels'], rc['masks'],
                            title=f'{mname} - {cat} - {CORRUPTION_LABELS[corr_name]} sev3',
                            save_path=os.path.join(FIG_DIR,
                                f'{cat}_{mname}_{corr_name}_sev3_heatmap.png'))
    return results


def run_augmented():
    results = {c: {m: {} for m in MODEL_NAMES} for c in CATEGORIES}

    for cat in CATEGORIES:
        print(f'\n{"="*60}\nCategory (augmented): {cat}\n{"="*60}')
        for mname in MODEL_NAMES:
            print(f'\n-- {mname} (augmented) --')
            model = build_model(mname)
            train_model(model, cat, augment=True)

            r = run_model_on(model, cat, tag='clean')
            results[cat][mname]['clean'] = {'img_auroc': r['img_auroc'],
                                            'pix_auroc': r['pix_auroc']}
            for corr_name in CORRUPTIONS:
                for sev in range(1, N_SEVERITIES + 1):
                    tag = f'{corr_name}_sev{sev}'
                    fn  = get_corruption_fn(corr_name, sev)
                    rc  = run_model_on(model, cat, corruption_fn=fn, tag=tag)
                    results[cat][mname][tag] = {'img_auroc': rc['img_auroc'],
                                                'pix_auroc': rc['pix_auroc']}
    return results


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f'  Saved {path}')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_csv(results, path):
    rows = []
    for cat, models in results.items():
        for model, conditions in models.items():
            for cond, m in conditions.items():
                rows.append({'category': cat, 'model': model,
                             'condition': cond,
                             'img_auroc': round(m['img_auroc'], 4),
                             'pix_auroc': round(m.get('pix_auroc', float('nan')), 4)})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['category','model','condition',
                                          'img_auroc','pix_auroc'])
        w.writeheader(); w.writerows(rows)
    print(f'  Saved {path}')


def print_clean_table(bl, aug):
    print(f'\n{"-"*65}')
    print('Clean Image-AUROC: Baseline vs Augmented')
    print(f'{"-"*65}')
    print(f'{"Category":<10} {"Model":<12} {"Baseline":>10} {"Augmented":>10} {"Δ":>8}')
    print('-'*65)
    for cat in CATEGORIES:
        for m in MODEL_NAMES:
            b = bl.get(cat,{}).get(m,{}).get('clean',{}).get('img_auroc', np.nan)
            a = aug.get(cat,{}).get(m,{}).get('clean',{}).get('img_auroc', np.nan)
            print(f'{cat:<10} {m:<12} {b:>10.4f} {a:>10.4f} {a-b:>+8.4f}')


def print_corruption_table(bl, aug):
    print(f'\n{"-"*82}')
    print('Avg Image-AUROC by Corruption (all severities & categories)')
    print(f'{"-"*82}')
    hdr = f'{"Corruption":<24}'
    for m in MODEL_NAMES:
        hdr += f'  {"BL-"+m:<13}{"Aug-"+m:<13}'
    print(hdr); print('-'*82)
    for corr in CORRUPTIONS:
        row = f'{CORRUPTION_LABELS[corr]:<24}'
        for m in MODEL_NAMES:
            bl_v, au_v = [], []
            for cat in CATEGORIES:
                for sev in range(1, N_SEVERITIES+1):
                    key = f'{corr}_sev{sev}'
                    bl_v.append(bl.get(cat,{}).get(m,{}).get(key,{}).get('img_auroc',np.nan))
                    au_v.append(aug.get(cat,{}).get(m,{}).get(key,{}).get('img_auroc',np.nan))
            row += f'  {np.nanmean(bl_v):<13.4f}{np.nanmean(au_v):<13.4f}'
        print(row)


def generate_all_figures(results_bl, results_aug=None):
    """Generate all aggregate figures from results dicts."""
    corr_names = list(CORRUPTIONS.keys())
    print('\nGenerating figures...')

    plot_auroc_vs_severity(
        results_bl, corr_names, MODEL_NAMES, metric='img_auroc',
        save_path=os.path.join(FIG_DIR, 'auroc_vs_severity_img.png'))
    print('  auroc_vs_severity_img.png')

    plot_auroc_vs_severity(
        results_bl, corr_names, MODEL_NAMES, metric='pix_auroc',
        save_path=os.path.join(FIG_DIR, 'auroc_vs_severity_pix.png'))
    print('  auroc_vs_severity_pix.png')

    plot_corruption_bar(
        results_bl, CATEGORIES, corr_names, MODEL_NAMES, metric='img_auroc',
        save_path=os.path.join(FIG_DIR, 'corruption_bar_img.png'))
    print('  corruption_bar_img.png')

    if results_aug is not None:
        plot_augmentation_comparison(
            results_bl, results_aug, CATEGORIES, corr_names, MODEL_NAMES,
            metric='img_auroc',
            save_path=os.path.join(FIG_DIR, 'augmentation_comparison.png'))
        print('  augmentation_comparison.png')

    print(f'Figures saved to: {FIG_DIR}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Anomaly detection robustness experiments')
    parser.add_argument(
        '--figures-only', action='store_true',
        help='Skip experiments; load existing JSON results and regenerate figures')
    parser.add_argument(
        '--phase', choices=['1', '2', '3', 'all'], default='all',
        help='Which phases to run (default: all). Ignored with --figures-only')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    t0 = time.time()
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        if args.figures_only:
            bl_path  = os.path.join(RESULTS_DIR, 'results_baseline.json')
            aug_path = os.path.join(RESULTS_DIR, 'results_augmented.json')
            if not os.path.exists(bl_path):
                print(f'ERROR: {bl_path} not found. Run experiments first.')
                sys.exit(1)
            print(f'Loading {bl_path}')
            results_bl  = load_json(bl_path)
            results_aug = load_json(aug_path) if os.path.exists(aug_path) else None
            if results_aug is None:
                print('No augmented results found; skipping augmentation comparison.')
            generate_all_figures(results_bl, results_aug)

        else:
            run_phase12 = args.phase in ('1', '2', 'all')
            run_phase3  = args.phase in ('3', 'all')

            if run_phase12:
                print('\n' + '='*60)
                print('PHASE 1 & 2: Baseline + Robustness Evaluation')
                print('='*60)
                results_bl = run_baseline_and_robustness()
                save_json(results_bl, os.path.join(RESULTS_DIR, 'results_baseline.json'))
                save_csv(results_bl,  os.path.join(RESULTS_DIR, 'results_baseline.csv'))
            else:
                results_bl = load_json(os.path.join(RESULTS_DIR, 'results_baseline.json'))

            results_aug = None
            if run_phase3:
                print('\n' + '='*60)
                print('PHASE 3: Corruption-Aware Augmentation')
                print('='*60)
                results_aug = run_augmented()
                save_json(results_aug, os.path.join(RESULTS_DIR, 'results_augmented.json'))
                save_csv(results_aug,  os.path.join(RESULTS_DIR, 'results_augmented.csv'))

            if results_aug is not None:
                print_clean_table(results_bl, results_aug)
                print_corruption_table(results_bl, results_aug)

            generate_all_figures(results_bl, results_aug)

        elapsed = time.time() - t0
        print(f'\nDone. Total time: {elapsed/60:.1f} min')
        sys.exit(0)

    except KeyboardInterrupt:
        print('\nInterrupted by user.')
        sys.exit(130)
    except Exception as e:
        print(f'\nERROR: {e}')
        raise
        sys.exit(1)

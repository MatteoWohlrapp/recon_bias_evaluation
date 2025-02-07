import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np


def performance_prediction(predictions, logger, results_dir):
    # Create a list to store all rows and create DataFrame at the end
    all_results = []
    classifier_metrics = {
        'ec' : "Enlarged Cardiomediastinum",
        'cardiomegaly': "Cardiomegaly",
        'lung-opacity': "Lung Opacity",
        'lung-lesion': "Lung Lesion",
        'edema': "Edema",
        'consolidation': "Consolidation",
        'pneumonia': "Pneumonia",
        'atelectasis': "Atelectasis",
        'pneumothorax': "Pneumothorax",
        'pleural-effusion': "Pleural Effusion",
        'pleural-other': "Pleural Other",
        'fracture': "Fracture",
    }
    image_metrics = {
        'psnr': 'psnr',
        'ssim': 'ssim',
        'nrmse': 'nrmse',
        'lpips': 'lpips',
    }



    for prediction in predictions:
        original_model = prediction['model']
        photon_count = prediction['photon_count']
        prediction_results = prediction['prediction_results']

        for metric, metric_name in image_metrics.items():
            value = prediction_results[metric_name].mean()

            # Add result to list
            all_results.append({
                'photon_count': photon_count,
                'model': original_model,
                'metric': metric,
                'value': value
            })
            
        baseline_auroc = []
        recon_auroc = []
        for metric, metric_name in classifier_metrics.items():

            # create mask for nan rows
            mask = prediction_results[metric_name].notna() & prediction_results[f'{metric_name}_class'].notna() & prediction_results[f'{metric_name}_recon'].notna()

            ground_truth = prediction_results[metric_name][mask]
            class_prediction = prediction_results[f'{metric_name}_class'][mask]
            recon_prediction = prediction_results[f'{metric_name}_recon'][mask]

            baseline_value = roc_auc_score(ground_truth, class_prediction)
            baseline_auroc.append(baseline_value)
            all_results.append({
                'photon_count': photon_count,
                'model': "baseline",
                'metric': metric,
                'value': baseline_value
            })

            recon_value = roc_auc_score(ground_truth, recon_prediction)
            recon_auroc.append(recon_value)
            all_results.append({
                'photon_count': photon_count,
                'model': original_model,
                'metric': metric,
                'value': recon_value
            })
        
        baseline_auroc = np.mean(baseline_auroc)
        all_results.append({
                'photon_count': photon_count,
                'model': "baseline",
                'metric': "average",
                'value': baseline_auroc
            })
        recon_auroc = np.mean(recon_auroc)
        all_results.append({
                'photon_count': photon_count,
                'model': original_model,
                'metric': "average",
                'value': recon_auroc
            })
    
    # Create final DataFrame from all results
    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def evaluate_chex(config, logger, results_dir, name):
    predictions = []

    for prediction in config['predictions']:
        model = prediction['model']
        photon_count = prediction['photon_count']
        prediction_results = pd.read_csv(prediction['predictions'])

        predictions.append({
            'model': model,
            'photon_count': photon_count,
            'prediction_results': prediction_results
        })

    evaluation_results = performance_prediction(predictions, logger, results_dir)
    evaluation_results.to_csv(results_dir / f'{name}_performance_results.csv', index=False)
        
        
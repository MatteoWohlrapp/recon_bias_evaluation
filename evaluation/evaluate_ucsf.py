import pandas as pd
from sklearn.metrics import roc_auc_score


def performance_prediction(predictions, logger, results_dir):
    # Create a list to store all rows and create DataFrame at the end
    all_results = []
    metrics = {
        'dice': 'UNet_dice', 
        'dice_recon': 'UNet_recon_dice',
        'sum_dice': 'UNet_sum',
        'sum_recon': 'UNet_recon_sum',
        'psnr': 'psnr',
        'ssim': 'ssim',
        'nrmse': 'nrmse',
        'lpips': 'lpips',
        'ttype': 'TTypeBCEClassifier',
        'ttype_recon': 'TTypeBCEClassifier_recon',
        'tgrade': 'TGradeBCEClassifier',
        'tgrade_recon': 'TGradeBCEClassifier_recon'
    }

    for prediction in predictions:
        original_model = prediction['model']
        acceleration = prediction['acceleration']
        prediction_results = prediction['prediction_results']

        for metric, metric_name in metrics.items():
            
            # For classification metrics, calculate AUROC
            if metric in ['ttype', 'ttype_recon']:
                # First aggregate predictions per patient using mean
                patient_preds = prediction_results.groupby('patient_id')[metric_name].median()
                # Get ground truth (one per patient)
                patient_gt = prediction_results.groupby('patient_id')['diagnosis'].first()
                # Calculate AUROC
                value = roc_auc_score(patient_gt, patient_preds)
            
            elif metric in ['tgrade', 'tgrade_recon']:
                # First aggregate predictions per patient using mean
                patient_preds = prediction_results.groupby('patient_id')[metric_name].median()
                # Get ground truth (one per patient)
                patient_gt = prediction_results.groupby('patient_id')['cns'].first()
                # Calculate AUROC
                value = roc_auc_score(patient_gt, patient_preds)
            
            else:
                # For other metrics, keep the original aggregation
                patient_results = prediction_results.groupby('patient_id')[metric_name]
                patient_values = patient_results.mean()
                value = patient_values.mean()

            current_model = "baseline" if metric in ["ttype", "tgrade", "dice"] else original_model

            if 'recon' in metric: 
                metric = metric.split('_')[0]

            # Add result to list
            all_results.append({
                'acceleration': acceleration,
                'model': current_model,
                'metric': metric,
                'value': value
            })
    
    # Create final DataFrame from all results
    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def evaluate_ucsf(config, logger, results_dir, name):
    predictions = []

    for prediction in config['predictions']:
        model = prediction['model']
        acceleration = prediction['acceleration']
        prediction_results = pd.read_csv(prediction['predictions'])

        predictions.append({
            'model': model,
            'acceleration': acceleration,
            'prediction_results': prediction_results
        })

    evaluation_results = performance_prediction(predictions, logger, results_dir)
    evaluation_results.to_csv(results_dir / f'{name}_performance_results.csv', index=False)
        
        
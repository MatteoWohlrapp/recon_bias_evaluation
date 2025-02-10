import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from plot_chex import plot_chex_performance, plot_chex_additional_bias
def get_tpr(y_pred, y):
    # calculate true positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    if y.sum() == 0:  # if no positive samples
        print(f"Warning: No positive samples found in ground truth. y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        print(f"y_pred values: {y_pred.value_counts()}")
        print(f"y values: {y.value_counts()}")
        return 0.0
    tpr = y_pred[y == 1].sum() / y.sum()
    return tpr

def get_fpr(y_pred, y):
    # calculate false positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    n_neg = len(y) - y.sum()
    if n_neg == 0:  # if no negative samples
        print(f"Warning: No negative samples found in ground truth. y_pred shape: {y_pred.shape}, y shape: {y.shape}")
        print(f"y_pred values: {y_pred.value_counts()}")
        print(f"y values: {y.value_counts()}")
        return 0.0
    fpr = y_pred[y == 0].sum() / n_neg
    return fpr

def calculate_threshold(y_pred, y_true):
    # Convert to numpy arrays to avoid pandas indexing issues
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # If all ground truth values are 0 or 1, use a reasonable default threshold
    if len(np.unique(y_true)) == 1:
        print(f"Warning: All ground truth values are {y_true[0]}. Using default threshold of 0.5")
        return 0.5
        
    # calculate threshold for the metric
    fpr, sens, threshs = roc_curve(y_true, y_pred)
    spec = 1 - fpr
    return threshs[np.argmin(np.abs(spec - sens))]

def fairness_prediction_classifier(predictions):
    predictions = predictions.copy()  # Make a shallow copy of the list

    sensitive_attributes = {
        'sex': (['Male', 'Female'], 'gender'),
        'age_bin': (["O", "Y"], 'age'), 
        'race': (["White", "Asian", "Other", "American Indian or Alaska Native", "Native Hawaiian or Other Pacific Islander", "Black"], 'ethnicity')
    }

    interpreters = {
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

    all_results = []
    for prediction in predictions:
        if prediction['fairness']:
            original_model = prediction['model']
            # Make a fresh copy for each interpreter to avoid data loss
            prediction_results = prediction['prediction_results'].copy()
            prediction_results['age_bin'] = prediction_results['age'].apply(lambda x: "Y" if float(str(x).split('(')[1].split(',')[0]) <= 62 else "O")

            average_EODD_class = {}
            average_EOP_class = {}
            average_EODD_recon = {}
            average_EOP_recon = {}
            average_delta_EODD_recon = {}
            average_delta_EOP_recon = {}
            for interpreter, interpreter_name in interpreters.items():
                # Create a filtered copy for this specific interpreter
                interpreter_results = prediction_results.copy()
                
                # Filter nan values for just this interpreter's columns
                mask = (
                    interpreter_results[f'{interpreter_name}'].notna() & 
                    interpreter_results[f'{interpreter_name}_class'].notna() & 
                    interpreter_results[f'{interpreter_name}_recon'].notna()
                )
                interpreter_results = interpreter_results[mask]

                baseline_threshold = calculate_threshold(interpreter_results[f'{interpreter_name}_class'], interpreter_results[f'{interpreter_name}'])
                recon_threshold = calculate_threshold(interpreter_results[f'{interpreter_name}_recon'], interpreter_results[f'{interpreter_name}'])

                # Store thresholded predictions back in the original prediction_results
                interpreter_results.loc[mask, f'{interpreter_name}_class'] = interpreter_results[f'{interpreter_name}_class'].apply(lambda x: 1 if x >= baseline_threshold else 0)
                interpreter_results.loc[mask, f'{interpreter_name}_recon'] = interpreter_results[f'{interpreter_name}_recon'].apply(lambda x: 1 if x >= recon_threshold else 0)

                for attribute, (attribute_values, attribute_name) in sensitive_attributes.items():
                    if not attribute_name in average_EODD_recon:
                        average_EODD_class[attribute_name] = []
                        average_EOP_class[attribute_name] = []
                        average_EODD_recon[attribute_name] = []
                        average_EOP_recon[attribute_name] = []
                        average_delta_EODD_recon[attribute_name] = []
                        average_delta_EOP_recon[attribute_name] = []

                    # calculate EODD, EOP, delta-EODD, delta-EOP
                    tpr_class = []
                    tpr_recon = []
                    fpr_class = []
                    fpr_recon = []

                    for attribute_value in attribute_values:
                        # Instead, create filtered copies:
                        filtered_pred_results = interpreter_results[interpreter_results[attribute] == attribute_value]

                        # Skip if no samples in this group
                        if len(filtered_pred_results) == 0:
                            print(f"Skipping {attribute}: {attribute_value} for {interpreter_name} - no samples found")
                            continue

                        # Skip if all samples are of one class
                        if len(filtered_pred_results[f'{interpreter_name}'].unique()) == 1:
                            print(f"Skipping {attribute}: {attribute_value} for {interpreter_name} - all samples are class {filtered_pred_results[f'{interpreter_name}'].iloc[0]}")
                            continue

                        tpr_class.append(get_tpr(filtered_pred_results[f'{interpreter_name}_class'], filtered_pred_results[f'{interpreter_name}']))
                        fpr_class.append(get_fpr(filtered_pred_results[f'{interpreter_name}_class'], filtered_pred_results[f'{interpreter_name}']))
                        tpr_recon.append(get_tpr(filtered_pred_results[f'{interpreter_name}_recon'], filtered_pred_results[f'{interpreter_name}']))
                        fpr_recon.append(get_fpr(filtered_pred_results[f'{interpreter_name}_recon'], filtered_pred_results[f'{interpreter_name}']))

                    # Only calculate metrics if we have at least two groups with valid metrics
                    if len(tpr_class) >= 2:
                        EODD_class = ((max(tpr_class) - min(tpr_class)) + (max(fpr_class) - min(fpr_class)))/2
                        EOP_class = (max(tpr_class) - min(tpr_class))
                        EODD_recon = ((max(tpr_recon) - min(tpr_recon)) + (max(fpr_recon) - min(fpr_recon)))/2
                        EOP_recon = (max(tpr_recon) - min(tpr_recon))

                        average_EODD_class[attribute_name].append(EODD_class)
                        average_EOP_class[attribute_name].append(EOP_class)
                        average_EODD_recon[attribute_name].append(EODD_recon)
                        average_EOP_recon[attribute_name].append(EOP_recon)
                        average_delta_EODD_recon[attribute_name].append(EODD_recon - EODD_class)
                        average_delta_EOP_recon[attribute_name].append(EOP_recon - EOP_class)

                        all_results.append({
                            'model': "baseline",
                            'interpreter': interpreter,
                            'attribute': attribute_name,
                            'metric': 'EODD',
                            'value': EODD_class,
                        })
                        all_results.append({
                            'model': "baseline",
                            'interpreter': interpreter,
                            'attribute': attribute_name,
                            'metric': 'EOP',
                            'value': EOP_class,
                        })

                        all_results.append({
                            'model': original_model,
                            'interpreter': interpreter,
                            'attribute': attribute_name,
                            'metric': 'EODD',
                            'value': EODD_recon,
                        })
                        all_results.append({
                            'model': original_model,
                            'interpreter': interpreter,
                            'attribute': attribute_name,
                            'metric': 'EOP',
                            'value': EOP_recon,
                        })
                        all_results.append({
                            'model': original_model,
                            'interpreter': interpreter,
                            'attribute': attribute_name,
                            'metric': 'delta-EODD',
                            'value': EODD_recon - EODD_class,
                        })
                        all_results.append({
                            'model': original_model,
                            'interpreter': interpreter,
                            'attribute': attribute_name,
                            'metric': 'delta-EOP',
                            'value': EOP_recon - EOP_class,
                        })

                    else:
                        print(f"Warning: Not enough valid groups for {attribute} to calculate fairness metrics")
                        continue
            
            for _, (_, attribute_name) in sensitive_attributes.items():
                all_results.append({
                    'model': "baseline",
                    'interpreter': 'average',
                    'attribute': attribute_name,
                    'metric': 'EODD',
                    'value': np.mean(average_EODD_class[attribute_name]),
                })
                all_results.append({
                    'model': "baseline",
                    'interpreter': 'average',
                    'attribute': attribute_name,
                    'metric': 'EOP',
                    'value': np.mean(average_EOP_class[attribute_name]),
                })
                all_results.append({
                    'model': original_model,
                    'interpreter': 'average',
                    'attribute': attribute_name,
                    'metric': 'EODD',
                    'value': np.mean(average_EODD_recon[attribute_name]),
                })
                all_results.append({
                    'model': original_model,
                    'interpreter': 'average',
                    'attribute': attribute_name,
                    'metric': 'EOP',
                    'value': np.mean(average_EOP_recon[attribute_name]),
                })
                all_results.append({
                    'model': original_model,
                    'interpreter': 'average',
                    'attribute': attribute_name,
                    'metric': 'delta-EODD',
                    'value': np.mean(average_delta_EODD_recon[attribute_name]),
                })
                all_results.append({
                    'model': original_model,
                    'interpreter': 'average',
                    'attribute': attribute_name,
                    'metric': 'delta-EOP',
                    'value': np.mean(average_delta_EOP_recon[attribute_name]),
                })

    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results

def performance_prediction(predictions):
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


def evaluate_chex(config, results_dir, name):
    predictions = []

    for prediction in config['predictions']:
        model = prediction['model']
        photon_count = prediction['photon_count']
        prediction_results = pd.read_csv(prediction['predictions'])
        fairness = prediction['fairness']

        predictions.append({
            'model': model,
            'photon_count': photon_count,
            'prediction_results': prediction_results,
            'fairness': fairness
        })

    performance_results = performance_prediction(predictions)
    performance_results.to_csv(results_dir / f'{name}_performance_results.csv', index=False)

    plot_chex_performance(performance_results, results_dir, name)

    fairness_results = fairness_prediction_classifier(predictions)
    fairness_results.to_csv(results_dir / f'{name}_fairness_results.csv', index=False)

    plot_chex_additional_bias(fairness_results, results_dir, name)
        
        
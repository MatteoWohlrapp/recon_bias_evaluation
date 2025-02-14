import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
from plot_ucsf import plot_ucsf_performance, plot_ucsf_additional_bias


def bootstrap_fairness(predictions, metric, attribute):
    pass


def get_tpr(y_pred, y):
    # calculate true positive rate
    y_pred = y_pred.astype(int)
    y = y.astype(int)
    if y.sum() == 0:  # if no positive samples
        print(
            f"Warning: No positive samples found in ground truth. y_pred shape: {y_pred.shape}, y shape: {y.shape}"
        )
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
        print(
            f"Warning: No negative samples found in ground truth. y_pred shape: {y_pred.shape}, y shape: {y.shape}"
        )
        print(f"y_pred values: {y_pred.value_counts()}")
        print(f"y values: {y.value_counts()}")
        return 0.0
    fpr = y_pred[y == 0].sum() / n_neg
    return fpr


def calculate_threshold(y_pred, y_true):
    # calculate threshold for the metric
    fpr, sens, threshs = roc_curve(y_true, y_pred)
    spec = 1 - fpr
    return threshs[np.argmin(np.abs(spec - sens))]


def fairness_prediction_classifier(predictions):
    predictions = predictions.copy()  # Make a shallow copy of the list

    sensitive_attributes = {
        "sex": (["M", "F"], "gender"),
        "age_bin": (["O", "Y"], "age"),
    }

    interpreters = {
        "ttype": ("TTypeBCEClassifier", "diagnosis"),
        "tgrade": ("TGradeBCEClassifier", "cns"),
    }

    all_results = []
    for prediction in predictions:
        # Make a deep copy of the prediction dictionary
        prediction = {
            "model": prediction["model"],
            "acceleration": prediction["acceleration"],
            "prediction_results": prediction["prediction_results"].copy(),
            "fairness": prediction["fairness"],
        }
        if prediction["fairness"]:
            original_model = prediction["model"]
            prediction_results = prediction["prediction_results"]
            prediction_results["age_bin"] = prediction_results["age"].apply(
                lambda x: (
                    "Y" if float(str(x).split("(")[1].split(",")[0]) <= 58 else "O"
                )
            )
            prediction_results["sex"] = prediction_results["sex"].apply(
                lambda x: "M" if float(str(x).split("(")[1].split(")")[0]) == 1 else "F"
            )
            prediction_results["cns"] = prediction_results["cns"].apply(
                lambda x: float(str(x).split("(")[1].split(")")[0])
            )
            prediction_results["diagnosis"] = prediction_results["diagnosis"].apply(
                lambda x: float(str(x).split("(")[1].split(")")[0])
            )

            for interpreter, (interpreter_name, gt_name) in interpreters.items():
                # Get patient-level predictions and include sensitive attributes in the groupby
                patient_preds_class = prediction_results.groupby("patient_id").agg(
                    {
                        f"{interpreter_name}": "median",
                        "sex": "first",
                        "age_bin": "first",
                    }
                )
                patient_preds_recon = prediction_results.groupby("patient_id").agg(
                    {
                        f"{interpreter_name}_recon": "median",
                        "sex": "first",
                        "age_bin": "first",
                    }
                )
                patient_gt = prediction_results.groupby("patient_id").agg(
                    {gt_name: "first", "sex": "first", "age_bin": "first"}
                )

                baseline_threshold = calculate_threshold(
                    patient_preds_class[f"{interpreter_name}"], patient_gt[gt_name]
                )
                recon_threshold = calculate_threshold(
                    patient_preds_recon[f"{interpreter_name}_recon"],
                    patient_gt[gt_name],
                )

                patient_preds_class[f"{interpreter_name}"] = patient_preds_class[
                    f"{interpreter_name}"
                ].apply(lambda x: 1 if x >= baseline_threshold else 0)
                patient_preds_recon[f"{interpreter_name}_recon"] = patient_preds_recon[
                    f"{interpreter_name}_recon"
                ].apply(lambda x: 1 if x >= recon_threshold else 0)

                for attribute, (
                    attribute_values,
                    attribute_name,
                ) in sensitive_attributes.items():
                    tpr_class = []
                    tpr_recon = []
                    fpr_class = []
                    fpr_recon = []

                    for attribute_value in attribute_values:
                        # Instead, create filtered copies:
                        filtered_preds_class = patient_preds_class[
                            patient_preds_class[attribute] == attribute_value
                        ]
                        filtered_preds_recon = patient_preds_recon[
                            patient_preds_recon[attribute] == attribute_value
                        ]
                        filtered_gt = patient_gt[
                            patient_gt[attribute] == attribute_value
                        ]

                        tpr_class.append(
                            get_tpr(
                                filtered_preds_class[f"{interpreter_name}"],
                                filtered_gt[gt_name],
                            )
                        )
                        fpr_class.append(
                            get_fpr(
                                filtered_preds_class[f"{interpreter_name}"],
                                filtered_gt[gt_name],
                            )
                        )
                        tpr_recon.append(
                            get_tpr(
                                filtered_preds_recon[f"{interpreter_name}_recon"],
                                filtered_gt[gt_name],
                            )
                        )
                        fpr_recon.append(
                            get_fpr(
                                filtered_preds_recon[f"{interpreter_name}_recon"],
                                filtered_gt[gt_name],
                            )
                        )

                    EODD_class = (
                        (max(tpr_class) - min(tpr_class))
                        + (max(fpr_class) - min(fpr_class))
                    ) / 2
                    EOP_class = max(tpr_class) - min(tpr_class)
                    EODD_recon = (
                        (max(tpr_recon) - min(tpr_recon))
                        + (max(fpr_recon) - min(fpr_recon))
                    ) / 2
                    EOP_recon = max(tpr_recon) - min(tpr_recon)

                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD",
                            "value": EODD_class,
                        }
                    )
                    all_results.append(
                        {
                            "model": "baseline",
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP",
                            "value": EOP_class,
                        }
                    )

                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EODD",
                            "value": EODD_recon,
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "EOP",
                            "value": EOP_recon,
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EODD",
                            "value": EODD_recon - EODD_class,
                        }
                    )
                    all_results.append(
                        {
                            "model": original_model,
                            "interpreter": interpreter,
                            "attribute": attribute_name,
                            "metric": "delta-EOP",
                            "value": EOP_recon - EOP_class,
                        }
                    )

    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def fairness_prediction_segmentation(predictions):
    predictions = predictions.copy()  # Make a shallow copy of the list

    sensitive_attributes = {
        "sex": (["M", "F"], "gender"),
        "age_bin": (["O", "Y"], "age"),
    }

    all_results = []
    for prediction in predictions:
        # Make a deep copy of the prediction dictionary
        prediction = {
            "model": prediction["model"],
            "acceleration": prediction["acceleration"],
            "prediction_results": prediction["prediction_results"].copy(),
        }
        original_model = prediction["model"]
        prediction_results = prediction["prediction_results"]
        prediction_results["age_bin"] = prediction_results["age"].apply(
            lambda x: "Y" if float(str(x).split("(")[1].split(",")[0]) <= 58 else "O"
        )
        prediction_results["sex"] = prediction_results["sex"].apply(
            lambda x: "M" if float(str(x).split("(")[1].split(")")[0]) == 1 else "F"
        )

        # Get patient-level predictions and include sensitive attributes in the groupby
        patient_preds_class = prediction_results.groupby("patient_id").agg(
            {"UNet_dice": "mean", "sex": "first", "age_bin": "first"}
        )
        patient_preds_recon = prediction_results.groupby("patient_id").agg(
            {"UNet_recon_dice": "mean", "sex": "first", "age_bin": "first"}
        )

        for attribute, (
            attribute_values,
            attribute_name,
        ) in sensitive_attributes.items():
            dice_class_by_group = []
            dice_recon_by_group = []

            for attribute_value in attribute_values:
                filtered_preds_class = patient_preds_class[
                    patient_preds_class[attribute] == attribute_value
                ]
                filtered_preds_recon = patient_preds_recon[
                    patient_preds_recon[attribute] == attribute_value
                ]

                dice_class_by_group.append(filtered_preds_class["UNet_dice"].mean())
                dice_recon_by_group.append(
                    filtered_preds_recon["UNet_recon_dice"].mean()
                )

            # Calculate SER for baseline (using 1 - Dice)
            ser_class = (1 - min(dice_class_by_group)) / (1 - max(dice_class_by_group))
            # Calculate delta between max and min Dice scores
            delta_dice_class = max(dice_class_by_group) - min(dice_class_by_group)

            # Calculate SER for reconstruction (using 1 - Dice)
            ser_recon = (1 - min(dice_recon_by_group)) / (1 - max(dice_recon_by_group))
            # Calculate delta between max and min Dice scores
            delta_dice_recon = max(dice_recon_by_group) - min(dice_recon_by_group)

            all_results.append(
                {
                    "model": "baseline",
                    "interpreter": "dice",
                    "attribute": attribute_name,
                    "metric": "SER",
                    "value": ser_class,
                }
            )
            all_results.append(
                {
                    "model": "baseline",
                    "interpreter": "dice",
                    "attribute": attribute_name,
                    "metric": "delta-dice",
                    "value": delta_dice_class,
                }
            )

            all_results.append(
                {
                    "model": original_model,
                    "interpreter": "dice",
                    "attribute": attribute_name,
                    "metric": "SER",
                    "value": ser_recon,
                }
            )
            all_results.append(
                {
                    "model": original_model,
                    "interpreter": "dice",
                    "attribute": attribute_name,
                    "metric": "delta-SER",
                    "value": ser_recon - ser_class,
                }
            )

            all_results.append(
                {
                    "model": original_model,
                    "interpreter": "dice",
                    "attribute": attribute_name,
                    "metric": "delta-dice",
                    "value": delta_dice_recon,
                }
            )

            all_results.append(
                {
                    "model": original_model,
                    "interpreter": "dice",
                    "attribute": attribute_name,
                    "metric": "delta-delta-dice",
                    "value": delta_dice_recon - delta_dice_class,
                }
            )

    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def psnr_difference_prediction(predictions):
    pass


def performance_prediction(predictions):
    predictions = predictions.copy()
    # Create a list to store all rows and create DataFrame at the end
    all_results = []
    metrics = {
        "dice": "UNet_dice",
        "dice_recon": "UNet_recon_dice",
        "sum_dice": "UNet_sum",
        "sum_recon": "UNet_recon_sum",
        "psnr": "psnr",
        "ssim": "ssim",
        "nrmse": "nrmse",
        "lpips": "lpips",
        "ttype": "TTypeBCEClassifier",
        "ttype_recon": "TTypeBCEClassifier_recon",
        "tgrade": "TGradeBCEClassifier",
        "tgrade_recon": "TGradeBCEClassifier_recon",
    }

    for prediction in predictions:
        original_model = prediction["model"]
        acceleration = prediction["acceleration"]
        prediction_results = prediction["prediction_results"]

        for metric, metric_name in metrics.items():

            # For classification metrics, calculate AUROC
            if metric in ["ttype", "ttype_recon"]:
                # First aggregate predictions per patient using mean
                patient_preds = prediction_results.groupby("patient_id")[
                    metric_name
                ].median()
                # Get ground truth (one per patient)
                patient_gt = prediction_results.groupby("patient_id")[
                    "diagnosis"
                ].first()
                # Calculate AUROC
                value = roc_auc_score(patient_gt, patient_preds)

            elif metric in ["tgrade", "tgrade_recon"]:
                # First aggregate predictions per patient using mean
                patient_preds = prediction_results.groupby("patient_id")[
                    metric_name
                ].median()
                # Get ground truth (one per patient)
                patient_gt = prediction_results.groupby("patient_id")["cns"].first()
                # Calculate AUROC
                value = roc_auc_score(patient_gt, patient_preds)

            else:
                # For other metrics, keep the original aggregation
                patient_results = prediction_results.groupby("patient_id")[metric_name]
                patient_values = patient_results.mean()
                value = patient_values.mean()

            current_model = (
                "baseline" if metric in ["ttype", "tgrade", "dice"] else original_model
            )

            if "recon" in metric:
                metric = metric.split("_")[0]

            # Add result to list
            all_results.append(
                {
                    "acceleration": acceleration,
                    "model": current_model,
                    "metric": metric,
                    "value": value,
                }
            )

    # Create final DataFrame from all results
    evaluation_results = pd.DataFrame(all_results)
    return evaluation_results


def evaluate_ucsf(config, results_dir, name):
    predictions = []

    for prediction in config["predictions"]:
        model = prediction["model"]
        acceleration = prediction["acceleration"]
        prediction_results = pd.read_csv(prediction["predictions"])
        fairness = prediction["fairness"]

        predictions.append(
            {
                "model": model,
                "acceleration": acceleration,
                "prediction_results": prediction_results,
                "fairness": fairness,
            }
        )

    performance_results = performance_prediction(predictions)
    performance_results.to_csv(
        results_dir / f"{name}_performance_results.csv", index=False
    )

    plot_ucsf_performance(performance_results, results_dir, name)

    fairness_results = fairness_prediction_classifier(predictions)
    fairness_results = pd.concat(
        [fairness_results, fairness_prediction_segmentation(predictions)]
    )
    fairness_results.to_csv(results_dir / f"{name}_fairness_results.csv", index=False)

    plot_ucsf_additional_bias(fairness_results, results_dir, name)

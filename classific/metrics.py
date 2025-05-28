# metrics.py

import logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm  # For progress bars during evaluation

import llm_wrappers  # To get predictions
import config  # For default metric type or other settings

logger = logging.getLogger(__name__)

# Define the set of possible labels your classifier LLM is expected to output
# This is important for consistent metric calculation, especially for confusion matrices
# and for ensuring sklearn metrics handle all potential classes correctly.
POSSIBLE_LABELS = ["SUCCESS", "FAILURE", "UNKNOWN"]

error_details = {}

def _get_predictions_for_prompt(prompt_text, eval_df, llm_task_model_name, max_samples_for_eval=None):
    """
    Gets predictions for a given prompt on an evaluation DataFrame.

    Args:
        prompt_text (str): The classifier prompt to evaluate.
        eval_df (pd.DataFrame): DataFrame with 'req', 'rsp', and 'label' columns.
        llm_task_model_name (str): Name of the LLM model to use for classification.
        max_samples_for_eval (int, optional): Maximum number of samples to evaluate.
                                              If None, evaluates on the whole DataFrame.

    Returns:
        tuple: (list_of_true_labels, list_of_predicted_labels, list_of_error_details)
               error_details is a list of dicts:
               [{'uuid': str, 'req': str, 'rsp': str, 'predicted_label': str, 'true_label': str}, ...]
    """
    true_labels = []
    predicted_labels = []
    error_details_list = []

    if not isinstance(eval_df, pd.DataFrame) or eval_df.empty:
        logger.warning("Evaluation DataFrame is empty or not a DataFrame. Cannot get predictions.")
        return true_labels, predicted_labels, error_details_list

    sample_df = eval_df
    # if max_samples_for_eval is not None and len(eval_df) > max_samples_for_eval:
    #     logger.info(f"Sampling {max_samples_for_eval} instances from eval_df (size {len(eval_df)}) for prediction.")
    #     sample_df = eval_df.sample(n=max_samples_for_eval,
    #                                random_state=config.random_seed if hasattr(config, 'random_seed') else None)

    # Use tqdm for a progress bar if evaluating many samples
    disable_tqdm = len(sample_df) < 10  # Don't show progress bar for very small sets

    for index, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Getting Predictions",
                           disable=disable_tqdm):
        req_text = str(row['req'])
        rsp_text = str(row['rsp'])
        true_label = str(row['label']).upper()  # Ensure consistent casing with POSSIBLE_LABELS

        if true_label not in POSSIBLE_LABELS:
            logger.warning(
                f"Encountered unexpected true label '{true_label}' for uuid {row.get('uuid', 'N/A')}. It might affect metric calculation if not in POSSIBLE_LABELS.")
            # Decide how to handle: skip, map to UNKNOWN, or include as is.
            # For now, include as is, but sklearn might warn or error if labels mismatch.

        try:
            predicted_label = llm_wrappers.classify_instance(
                classifier_prompt_text=prompt_text,
                req_text=req_text,
                rsp_text=rsp_text,
                model_name=llm_task_model_name
            )
            # classify_instance should return one of POSSIBLE_LABELS or handle its own parsing
        except Exception as e:
            logger.error(f"Error classifying instance (uuid: {row.get('uuid', 'N/A')}): {e}. Assigning 'UNKNOWN'.")
            predicted_label = "UNKNOWN"  # Default prediction on error

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        if predicted_label != true_label:
            error_details_list.append({
                'uuid': str(row.get('uuid', f'index_{index}')),  # Use UUID if available, else index
                'req': req_text,
                'rsp': rsp_text,
                'predicted_label': predicted_label,
                'true_label': true_label
            })

    return true_labels, predicted_labels, error_details_list


def calculate_metric_for_prompt(prompt_text, eval_df, llm_task_model_name, metric_type, max_samples_for_eval=None):
    """
    Calculates a specified metric for a given prompt on an evaluation DataFrame.

    Args:
        prompt_text (str): The classifier prompt to evaluate.
        eval_df (pd.DataFrame): DataFrame with 'req', 'rsp', and 'label' columns.
        llm_task_model_name (str): Name of the LLM model to use for classification.
        metric_type (str): The metric to calculate (e.g., 'accuracy', 'f1_macro', 'f1_micro',
                           'f1_weighted', 'precision_macro', 'recall_macro').
        max_samples_for_eval (int, optional): Max samples to use for evaluation. If None, uses all.

    Returns:
        float: The calculated metric score. Returns -1.0 on error or if no predictions.
    """
    logger.debug(f"Calculating metric '{metric_type}' for prompt: {prompt_text[:70]}...")

    true_labels, predicted_labels, error_details_list = _get_predictions_for_prompt(
        prompt_text, eval_df, llm_task_model_name, max_samples_for_eval
    )

    if not true_labels or not predicted_labels:
        logger.warning("No true or predicted labels obtained. Cannot calculate metric.")
        return -1.0  # Indicate error or inability to score

    try:
        if metric_type == 'accuracy':
            score = accuracy_score(true_labels, predicted_labels)
        elif metric_type in ['f1_macro', 'f1_micro', 'f1_weighted',
                             'precision_macro', 'precision_micro', 'precision_weighted',
                             'recall_macro', 'recall_micro', 'recall_weighted']:
            avg_type = metric_type.split('_')[-1]  # macro, micro, or weighted
            metric_name = metric_type.split('_')[0]  # f1, precision, or recall

            # Ensure labels argument includes all possible classes for robust averaging,
            # especially if some classes don't appear in true_labels or predicted_labels for a small sample.
            p, r, f1, _ = precision_recall_fscore_support(
                true_labels,
                predicted_labels,
                average=avg_type,
                labels=POSSIBLE_LABELS,  # Use predefined labels
                zero_division=0  # Return 0 if a class has no predictions/true instances
            )
            if metric_name == 'f1':
                score = f1
            elif metric_name == 'precision':
                score = p
            elif metric_name == 'recall':
                score = r
            else:
                logger.error(f"Unsupported detailed metric name: {metric_name}")
                return -1.0
        else:
            logger.error(f"Unsupported metric_type: {metric_type}")
            return -1.0

        logger.debug(f"Metric '{metric_type}' calculated: {score:.4f}")
        return score
    except Exception as e:
        logger.error(f"Error calculating metric '{metric_type}': {e}")
        logger.error(f"True labels ({len(true_labels)}): {true_labels[:10]}")
        logger.error(f"Predicted labels ({len(predicted_labels)}): {predicted_labels[:10]}")
        # You might want to log the confusion matrix here for debugging
        try:
            cm = confusion_matrix(true_labels, predicted_labels, labels=POSSIBLE_LABELS)
            logger.error(f"Confusion Matrix (Rows: True, Cols: Pred, Labels: {POSSIBLE_LABELS}):\n{cm}")
        except Exception as cm_e:
            logger.error(f"Could not generate confusion matrix: {cm_e}")
        return -1.0


def get_errors_for_prompt(prompt_text, eval_df, llm_task_model_name, max_samples_for_errors=None):
    """
    Identifies instances where the prompt makes errors on the evaluation DataFrame.

    Args:
        prompt_text (str): The classifier prompt.
        eval_df (pd.DataFrame): DataFrame with 'req', 'rsp', and 'label' columns.
        llm_task_model_name (str): Name of the LLM model for classification.
        max_samples_for_errors (int, optional): Max samples to check for errors.
                                                If None, checks the whole DataFrame.
                                                This is different from max_samples_for_eval in
                                                calculate_metric, as this is specifically for error mining.

    Returns:
        list: A list of dictionaries, where each dictionary details an error instance:
              {'uuid': str, 'req': str, 'rsp': str, 'predicted_label': str, 'true_label': str}
    """
    logger.debug(f"Getting error details for prompt: {prompt_text[:70]}...")

    # _get_predictions_for_prompt already filters by max_samples if provided
    # So, max_samples_for_errors is passed as max_samples_for_eval to it.
    _, _, error_details_list = _get_predictions_for_prompt(
        prompt_text, eval_df, llm_task_model_name, max_samples_for_errors
    )

    logger.info(f"Found {len(error_details_list)} errors for the prompt on the evaluated sample.")
    return error_details_list


def get_detailed_classification_report(prompt_text, eval_df, llm_task_model_name, max_samples_for_report=None):
    """
    Generates a more detailed classification report including confusion matrix
    and per-class precision, recall, F1-score.

    Args:
        prompt_text (str): The classifier prompt.
        eval_df (pd.DataFrame): DataFrame with 'req', 'rsp', and 'label' columns.
        llm_task_model_name (str): Name of the LLM model for classification.
        max_samples_for_report (int, optional): Max samples to use for the report.

    Returns:
        dict: A dictionary containing 'accuracy', 'confusion_matrix',
              'per_class_metrics' (a dict per class), and 'overall_metrics' (macro/weighted).
              Returns None on error.
    """
    logger.info(f"Generating detailed classification report for prompt: {prompt_text[:70]}...")

    true_labels, predicted_labels, _ = _get_predictions_for_prompt(
        prompt_text, eval_df, llm_task_model_name, max_samples_for_report
    )

    if not true_labels or not predicted_labels:
        logger.warning("No true or predicted labels obtained. Cannot generate detailed report.")
        return None

    report = {}
    try:
        report['accuracy'] = accuracy_score(true_labels, predicted_labels)

        # Confusion Matrix
        # Ensure labels are consistently ordered using POSSIBLE_LABELS
        cm = confusion_matrix(true_labels, predicted_labels, labels=POSSIBLE_LABELS)
        report['confusion_matrix'] = cm.tolist()  # Convert numpy array to list for easier serialization
        report['confusion_matrix_labels'] = POSSIBLE_LABELS

        # Per-class and overall metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=POSSIBLE_LABELS,
            zero_division=0
        )

        per_class_metrics = {}
        for i, label in enumerate(POSSIBLE_LABELS):
            per_class_metrics[label] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1-score': f1[i],
                'support': int(support[i])  # Convert numpy int to Python int
            }
        report['per_class_metrics'] = per_class_metrics

        # Overall (macro and weighted)
        report['overall_metrics'] = {}
        for avg_type in ['macro', 'weighted']:  # Micro is often same as accuracy for multiclass
            p_avg, r_avg, f1_avg, _ = precision_recall_fscore_support(
                true_labels,
                predicted_labels,
                average=avg_type,
                labels=POSSIBLE_LABELS,
                zero_division=0
            )
            report['overall_metrics'][avg_type] = {
                'precision': p_avg,
                'recall': r_avg,
                'f1-score': f1_avg
            }

        logger.info("Detailed classification report generated successfully.")
        return report

    except Exception as e:
        logger.error(f"Error generating detailed classification report: {e}")
        return None


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    logger.info("Testing metrics.py...")

    # Mock llm_wrappers.classify_instance for testing without actual API calls
    # In a real test, you might want to use a mock library like unittest.mock
    MOCK_PREDICTIONS = {}  # Store mock predictions: (prompt_hash, req, rsp) -> label


    def mock_classify_instance(classifier_prompt_text, req_text, rsp_text, model_name):
        # Simple mock: if req contains "success", predict SUCCESS, else if "fail", predict FAILURE
        # This is a very basic mock. A better mock might use a hash of the prompt.
        logger.debug(f"Mock classify_instance called for prompt: {classifier_prompt_text[:30]}...")
        if "make this succeed" in classifier_prompt_text.lower():
            if "good" in req_text.lower(): return "SUCCESS"
            if "bad" in req_text.lower(): return "FAILURE"
        if "make this fail" in classifier_prompt_text.lower():  # A "worse" prompt
            if "good" in req_text.lower(): return "FAILURE"  # Intentionally wrong
            if "bad" in req_text.lower(): return "SUCCESS"  # Intentionally wrong

        # Default mock behavior
        if "good data" in req_text.lower() and "200 ok" in rsp_text.lower():
            return "SUCCESS"
        elif "bad data" in req_text.lower() or "500 error" in rsp_text.lower():
            return "FAILURE"
        else:
            return "UNKNOWN"


    original_classify_instance = llm_wrappers.classify_instance
    llm_wrappers.classify_instance = mock_classify_instance  # Monkey patch

    # Create dummy config attributes if not present
    if not hasattr(config, 'random_seed'): config.random_seed = 42
    if not hasattr(config, 'llm_task_model_name'): config.llm_task_model_name = "mock_model"

    # Create dummy evaluation data
    data = {
        'uuid': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7'],
        'req': ["good data request", "bad data request", "ambiguous request", "good data again", "another bad data",
                "neutral req", "good req"],
        'rsp': ["200 OK response", "500 error response", "generic ack", "200 OK", "failed attempt", "202 Accepted",
                "200 OK"],
        'label': ["SUCCESS", "FAILURE", "UNKNOWN", "SUCCESS", "FAILURE", "UNKNOWN", "SUCCESS"]
    }
    test_df = pd.DataFrame(data)

    test_prompt_good = "This is a good prompt. If req has 'good data' and rsp has '200 OK', it's SUCCESS. If 'bad data' or '500 error', it's FAILURE. Otherwise UNKNOWN. Make this succeed."
    test_prompt_bad = "This is a bad prompt that often gets things wrong. Make this fail."

    # Test 1: calculate_metric_for_prompt
    logger.info("\n--- Testing calculate_metric_for_prompt ---")
    accuracy_good = calculate_metric_for_prompt(test_prompt_good, test_df, config.llm_task_model_name, 'accuracy')
    logger.info(f"Accuracy for good prompt: {accuracy_good:.4f}")

    f1_macro_good = calculate_metric_for_prompt(test_prompt_good, test_df, config.llm_task_model_name, 'f1_macro')
    logger.info(f"F1 Macro for good prompt: {f1_macro_good:.4f}")

    accuracy_bad = calculate_metric_for_prompt(test_prompt_bad, test_df, config.llm_task_model_name, 'accuracy')
    logger.info(f"Accuracy for bad prompt: {accuracy_bad:.4f}")

    f1_macro_bad = calculate_metric_for_prompt(test_prompt_bad, test_df, config.llm_task_model_name, 'f1_macro')
    logger.info(f"F1 Macro for bad prompt: {f1_macro_bad:.4f}")

    # Test 2: get_errors_for_prompt
    logger.info("\n--- Testing get_errors_for_prompt ---")
    errors_good_prompt = get_errors_for_prompt(test_prompt_good, test_df, config.llm_task_model_name)
    logger.info(f"Errors for good prompt ({len(errors_good_prompt)}):")
    for err in errors_good_prompt:
        logger.info(
            f"  UUID: {err['uuid']}, Req: {err['req'][:20]}..., Pred: {err['predicted_label']}, True: {err['true_label']}")

    errors_bad_prompt = get_errors_for_prompt(test_prompt_bad, test_df, config.llm_task_model_name)
    logger.info(f"Errors for bad prompt ({len(errors_bad_prompt)}):")
    for err in errors_bad_prompt:
        logger.info(
            f"  UUID: {err['uuid']}, Req: {err['req'][:20]}..., Pred: {err['predicted_label']}, True: {err['true_label']}")

    # Test 3: get_detailed_classification_report
    logger.info("\n--- Testing get_detailed_classification_report ---")
    report_good = get_detailed_classification_report(test_prompt_good, test_df, config.llm_task_model_name)
    if report_good:
        logger.info(f"Detailed report for good prompt:")
        logger.info(f"  Accuracy: {report_good['accuracy']:.4f}")
        logger.info(
            f"  Confusion Matrix (Labels: {report_good['confusion_matrix_labels']}):\n{pd.DataFrame(report_good['confusion_matrix'], index=report_good['confusion_matrix_labels'], columns=report_good['confusion_matrix_labels'])}")
        logger.info(f"  Per-class metrics: {report_good['per_class_metrics']}")
        logger.info(f"  Overall metrics: {report_good['overall_metrics']}")

    # Restore original function if monkey patched
    llm_wrappers.classify_instance = original_classify_instance
    logger.info("\n--- metrics.py tests finished ---")


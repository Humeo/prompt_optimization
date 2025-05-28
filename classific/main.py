import argparse
import logging
import os
import pandas as pd
import json
import time
import sys  # For sys.exit on critical errors

# Custom modules
import config  # Project configuration
import data_utils
import protegi_algorithm
import metrics
import llm_wrappers  # To ensure it's loaded and configured (e.g., API keys)

# --- Logger Setup ---
# Configure root logger to capture logs from all modules
# We will add a file handler later based on output_dir
# Basic configuration will be updated after parsing args for log level and output dir
logging.basicConfig(
    level=logging.INFO,  # Default level, can be overridden
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # Log to console by default
)
logger = logging.getLogger(__name__)  # Logger for this main script


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the ProTeGi prompt optimization algorithm.")

    # Optional overrides for config parameters
    # Get defaults from config module if attributes exist, otherwise show N/A
    def get_config_default(attr, default_val="N/A"):
        return getattr(config, attr, default_val)

# --train_data "demo_data/labeled_demo_data_aitmg_202504.csv" --test_data "demo_data/labeled_demo_data_aitmg_202504.csv" --initial_prompt_file "prompts/initial_classifier_prompt.txt"

    parser.add_argument("--train_data", type=str, default="demo_data/labeled_demo_data_aitmg_202504.csv",
                        help="Path to the training data CSV file (req, rsp, label, uuid).")
    parser.add_argument("--test_data", type=str, default="demo_data/labeled_demo_data_aitmg_202504.csv",
                        help="Path to the test data CSV file (req, rsp, label, uuid).")
    parser.add_argument("--initial_prompt_file", type=str, default="prompts/initial_classifier_prompt.txt",
                        help="Path to the file containing the initial prompt text.")
    parser.add_argument("--output_dir", type=str, default="protegi_results",
                        help="Directory to save results, logs, and the best prompt.")


    parser.add_argument("--search_depth", type=int, default=config.search_depth,
                        help=f"Override config.search_depth (default: {get_config_default('search_depth')}).")
    parser.add_argument("--beam_width", type=int, default=config.beam_width,
                        help=f"Override config.beam_width (default: {get_config_default('beam_width')}).")
    parser.add_argument("--llm_task_model_name", type=str,  default=config.llm_task_model_name,
                        help=f"Override config.llm_task_model_name (default: {get_config_default('llm_task_model_name')}).")
    parser.add_argument("--metric_type", type=str, default="accuracy",
                        help=f"Override config.metric_type (default: {get_config_default('metric_type')}).")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO).")
    parser.add_argument("--max_samples_for_final_eval", type=int, default=5,
                        help=f"Max samples for final evaluation on test set (default: {get_config_default('max_samples_for_final_eval', 'None (all)')}).")

    args = parser.parse_args()
    return args


def apply_config_overrides(args, config_module):
    """Apply command-line arguments to override config module attributes."""
    overridden_params = {}
    if args.search_depth is not None:
        config_module.search_depth = args.search_depth
        overridden_params["search_depth"] = config_module.search_depth
    if args.beam_width is not None:
        config_module.beam_width = args.beam_width
        overridden_params["beam_width"] = config_module.beam_width
    if args.llm_task_model_name is not None:
        config_module.llm_task_model_name = args.llm_task_model_name
        overridden_params["llm_task_model_name"] = config_module.llm_task_model_name
    if args.metric_type is not None:
        config_module.metric_type = args.metric_type
        overridden_params["metric_type"] = config_module.metric_type
    if args.max_samples_for_final_eval is not None:
        config_module.max_samples_for_final_eval = args.max_samples_for_final_eval
        overridden_params["max_samples_for_final_eval"] = config_module.max_samples_for_final_eval

    if overridden_params:
        logger.info(f"Applied CLI overrides to config: {overridden_params}")

    # Update logging level for all loggers
    log_level_numeric = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_numeric)  # Set level on root logger
    for handler in logging.getLogger().handlers:  # Also set on existing handlers
        handler.setLevel(log_level_numeric)
    logger.info(f"Logging level set to: {args.log_level.upper()}")


def load_initial_prompt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.critical(f"Initial prompt file not found: {filepath}")
        raise
    except Exception as e:
        logger.critical(f"Error loading initial prompt from {filepath}: {e}")
        raise


def save_results(output_dir, best_prompt, initial_results, final_results, experiment_params):
    """Saves the best prompt and evaluation results."""
    os.makedirs(output_dir, exist_ok=True)

    # Save best prompt
    best_prompt_path = os.path.join(output_dir, "best_prompt_found.txt")
    with open(best_prompt_path, 'w', encoding='utf-8') as f:
        f.write(best_prompt if best_prompt else "ERROR: No best prompt was determined.")
    logger.info(f"Best prompt saved to: {best_prompt_path}")

    # Save results summary
    results_summary = {
        "experiment_parameters": experiment_params,
        "initial_prompt_evaluation_on_test": initial_results if initial_results else "Evaluation failed or not performed.",
        "optimized_prompt_evaluation_on_test": final_results if final_results else "Evaluation failed or not performed.",
        "optimized_prompt_text_path": "best_prompt_found.txt"  # Relative path
    }
    results_path = os.path.join(output_dir, "experiment_summary.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    logger.info(f"Experiment summary saved to: {results_path}")


# --- Main Execution ---
def main():
    start_time = time.time()
    args = parse_arguments()

    # --- Setup Output Directory and File Logger ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(args.output_dir, f"protegi_run_{timestamp}.log")

    # Remove default StreamHandler if we are setting new ones or to avoid duplicate console logs
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:  # Iterate over a copy
        root_logger.removeHandler(handler)
        handler.close()

    # Add new handlers: one for console, one for file
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Apply config overrides from CLI arguments (this also sets the log level)
    apply_config_overrides(args, config)

    logger.info("======================================================================")
    logger.info("ProTeGi Experiment Run Started")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Log file: {os.path.abspath(log_file_path)}")
    logger.info(f"Command line arguments: {vars(args)}")
    logger.info("======================================================================")

    # --- Load Data ---
    logger.info("Loading data...")
    try:
        # Expecting columns: 'uuid', 'req', 'rsp', 'label'
        Dtr_train_df = data_utils.load_data(args.train_data,)
        Dte_test_df = data_utils.load_data(args.test_data,)
        initial_prompt = load_initial_prompt(args.initial_prompt_file)
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load data or initial prompt. Exiting. Error: {e}", exc_info=True)
        sys.exit(1)  # Exit if essential data is missing

    if Dtr_train_df.empty:
        logger.critical("Training data is empty. Exiting.")
        sys.exit(1)
    if Dte_test_df.empty:
        logger.critical("Test data is empty. Exiting.")
        sys.exit(1)
    if not initial_prompt:  # load_initial_prompt would have raised, but as a safeguard
        logger.critical("Initial prompt is empty. Exiting.")
        sys.exit(1)

    logger.info(f"Training data loaded: {len(Dtr_train_df)} samples from {args.train_data}")
    logger.info(f"Test data loaded: {len(Dte_test_df)} samples from {args.test_data}")
    logger.info(
        f"Initial prompt loaded from {args.initial_prompt_file}: {initial_prompt[:150].replace(os.linesep, ' ')}...")

    # --- Log key configuration parameters being used ---
    # Ensure all expected config attributes are present or have defaults
    def get_cfg(attr, default=None):
        return getattr(config, attr, default)

    experiment_params = {
        "timestamp": timestamp,
        "cli_args": vars(args),
        "effective_config": {
            "train_data_path": args.train_data,
            "test_data_path": args.test_data,
            "initial_prompt_file": args.initial_prompt_file,
            "output_dir": args.output_dir,
            "search_depth": get_cfg('search_depth'),
            "beam_width": get_cfg('beam_width'),
            "metric_type": get_cfg('metric_type'),
            "selection_strategy": get_cfg('selection_strategy', "fallback"),  # Default if not in config
            "llm_task_model": get_cfg('llm_task_model_name'),
            "llm_gradient_model": get_cfg('llm_gradient_model_name'),
            "llm_edit_model": get_cfg('llm_edit_model_name'),
            "llm_paraphrase_model": get_cfg('llm_paraphrase_model_name'),
            "minibatch_size_for_errors": get_cfg('minibatch_size_for_errors'),
            "max_error_examples_for_gradient": get_cfg('max_error_examples_for_gradient'),
            "num_gradients_to_generate": get_cfg('num_gradients_to_generate'),
            "num_edits_per_gradient": get_cfg('num_edits_per_gradient'),
            "num_paraphrases_to_generate": get_cfg('num_paraphrases_to_generate'),
            "fallback_eval_size": get_cfg('fallback_eval_size'),
            "max_samples_for_final_eval": get_cfg('max_samples_for_final_eval', None)  # Default to None if not set
        }
    }
    logger.info(f"Running with effective parameters: {json.dumps(experiment_params['effective_config'], indent=2)}")

    # --- Evaluate Initial Prompt on Test Set ---
    logger.info("\n--- Evaluating Initial Prompt on Test Set ---")
    initial_prompt_results_on_test = None
    # try:
    #     initial_prompt_results_on_test = metrics.get_detailed_classification_report(
    #         prompt_text=initial_prompt,
    #         eval_df=Dte_test_df,
    #         llm_task_model_name=config.llm_task_model_name,
    #         max_samples_for_report=config.max_samples_for_final_eval
    #     )
    #     if initial_prompt_results_on_test:
    #         logger.info(f"Initial prompt test set metric ({config.metric_type}): "
    #                     f"{initial_prompt_results_on_test.get('overall_metrics', {}).get(config.metric_type.split('_')[-1] if '_' in config.metric_type else 'macro', {}).get(config.metric_type.split('_')[0] if '_' in config.metric_type else config.metric_type, 'N/A')}")
    #         logger.info(f"Initial prompt test set accuracy: {initial_prompt_results_on_test.get('accuracy', 'N/A')}")
    #         logger.info(
    #             f"Initial prompt test set detailed report:\n{json.dumps(initial_prompt_results_on_test, indent=2)}")
    #     else:
    #         logger.warning("Failed to get detailed report for the initial prompt on the test set.")
    # except Exception as e:
    #     logger.error(f"Error evaluating initial prompt on test set: {e}", exc_info=True)
    #     initial_prompt_results_on_test = {"error": str(e)}

    # --- Run ProTeGi Algorithm ---
    logger.info("\n--- Running ProTeGi Algorithm ---")
    try:
        best_prompt_found = protegi_algorithm.run_protegi(
            initial_prompt=initial_prompt,
            Dtr_train_df=Dtr_train_df,
            config_obj=config  # Pass the whole config module
        )
    except Exception as e:
        logger.critical(f"CRITICAL: ProTeGi algorithm failed. Error: {e}", exc_info=True)
        best_prompt_found = None  # Indicate failure
        # Optionally, save whatever partial results or state might be useful
        save_results(args.output_dir, f"ERROR_DURING_PROTEGI: {e}", initial_prompt_results_on_test,
                     {"error": f"ProTeGi failed: {e}"}, experiment_params)
        sys.exit(1)

    if not best_prompt_found:
        logger.error("ProTeGi algorithm did not return a best prompt. Using initial prompt for final evaluation.")
        best_prompt_found = initial_prompt  # Fallback to initial if ProTeGi fails to return one

    logger.info(f"\nProTeGi finished. Best prompt found:\n{best_prompt_found}")

    # --- Evaluate Best Found Prompt on Test Set ---
    logger.info("\n--- Evaluating Best Found Prompt on Test Set ---")
    final_prompt_results_on_test = None
    if best_prompt_found:
        try:
            final_prompt_results_on_test = metrics.get_detailed_classification_report(
                prompt_text=best_prompt_found,
                eval_df=Dte_test_df,
                llm_task_model_name=config.llm_task_model_name,
                max_samples_for_report=config.max_samples_for_final_eval
            )
            if final_prompt_results_on_test:
                logger.info(f"Optimized prompt test set metric ({config.metric_type}): "
                            f"{final_prompt_results_on_test.get('overall_metrics', {}).get(config.metric_type.split('_')[-1] if '_' in config.metric_type else 'macro', {}).get(config.metric_type.split('_')[0] if '_' in config.metric_type else config.metric_type, 'N/A')}")
                logger.info(
                    f"Optimized prompt test set accuracy: {final_prompt_results_on_test.get('accuracy', 'N/A')}")
                logger.info(
                    f"Optimized prompt test set detailed report:\n{json.dumps(final_prompt_results_on_test, indent=2)}")
            else:
                logger.warning("Failed to get detailed report for the optimized prompt on the test set.")
        except Exception as e:
            logger.error(f"Error evaluating best found prompt on test set: {e}", exc_info=True)
            final_prompt_results_on_test = {"error": str(e)}
    else:
        logger.warning("No best prompt was identified by ProTeGi. Skipping final evaluation of optimized prompt.")

    # --- Save Results ---
    logger.info("\n--- Saving Results ---")
    save_results(args.output_dir, best_prompt_found, initial_prompt_results_on_test, final_prompt_results_on_test,
                 experiment_params)

    end_time = time.time()
    total_duration = end_time - start_time
    logger.info(f"\nProTeGi Experiment Run Finished. Total duration: {total_duration:.2f} seconds.")
    logger.info("======================================================================")


if __name__ == "__main__":
    # This ensures that llm_wrappers (and thus API key loading) is handled early
    # if there are any global setups in llm_wrappers.
    # try:
    #     llm_wrappers.ensure_client_setup() # Example: if you have a function to init clients
    #     logger.info("LLM clients initialized (if applicable).")
    # except AttributeError:
    #     logger.debug("No 'ensure_client_setup' found in llm_wrappers, or not needed.")
    # except Exception as e:
    #     logger.warning(f"Could not perform initial LLM client setup: {e}")

    main()

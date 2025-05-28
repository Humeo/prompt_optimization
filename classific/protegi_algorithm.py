# protegi_algorithm.py
import json
import logging
import random
import math  # For UCB or other selection strategies
import time

import llm_wrappers
import data_utils
import metrics
import config  # Your project's configuration file

logger = logging.getLogger(__name__)


def _get_error_examples_string(prompt_text, Dmini_df, llm_task_model_name, max_errors_for_gradient=10):
    """
    Runs the given prompt on a minibatch Dmini and returns a string
    representation of error examples.
    """
    error_instances = metrics.get_errors_for_prompt(
        prompt_text,
        Dmini_df,
        llm_task_model_name,
        None  # This might not be directly used by get_errors_for_prompt but good to be aware
    )

    if not error_instances:
        return ""

    # Shuffle and limit the number of error examples to show to the LLM_gradient
    random.shuffle(error_instances)
    error_examples_str_list = []
    for i, error in enumerate(error_instances[:max_errors_for_gradient]):
        # error is expected to be a dict like {'uuid': ..., 'req': ..., 'rsp': ..., 'predicted_label': ..., 'true_label': ...}
        error_str = (
            # f"Error Example {i + 1}:\n"
            # f"  Request: {error['req'][:config.max_chars_per_error_example_field] if hasattr(config, 'max_chars_per_error_example_field') else error['req'][:200]}...\n"
            # f"  Response: {error['rsp'][:config.max_chars_per_error_example_field] if hasattr(config, 'max_chars_per_error_example_field') else error['rsp'][:200]}...\n"
            # f"  Predicted Label: {error['predicted_label']}\n"
            # f"  Correct Label: {error['true_label']}\n"
            # f"---"

            f"Error Example {i + 1}:\n"
            f"  [REQ START]\n"
            f"  {error['req']}\n"
            f"  [REQ END]\n"
            f"  [RSP START]\n"
            f"  Response: {error['rsp']}\n"
            f"  [RSP END]\n"
            f"  Predicted Label: {error['predicted_label']}\n"
            f"  Correct Label: {error['true_label']}\n"
            f"---"
        )
        error_examples_str_list.append(error_str)

    return "\n".join(error_examples_str_list)


def expand_single_prompt(prompt_candidate, Dtr_train_df, config_obj):
    """
    Expands a single prompt candidate by:
    1. Generating textual gradients based on errors on a minibatch.
    2. Editing the prompt using these gradients.
    3. Paraphrasing the prompt.

    Args:
        prompt_candidate (str): The prompt to expand.
        Dtr_train_df (pd.DataFrame): The training dataset to sample minibatches from.
        config_obj (module): The configuration module.

    Returns:
        list: A list of new prompt strings generated from this candidate.
    """
    newly_generated_prompts = []

    # 1. Sample Dmini and find errors for gradient generation
    if Dtr_train_df.empty:
        logger.warning(
            f"Training data for Dmini is too small ({len(Dtr_train_df)}) or empty. Skipping gradient generation for: {prompt_candidate[:50]}...")
        error_examples_str = ""
    else:
        Dmini = data_utils.sample_minibatch(
            Dtr_train_df,
            min(config_obj.minibatch_size_for_errors, len(Dtr_train_df)),
            stratify_col='label'  # Attempt stratification
        )
        if Dmini.empty:
            logger.warning(f"Sampled Dmini is empty. Skipping gradient generation for: {prompt_candidate[:50]}...")
            error_examples_str = ""
        else:
            logger.info(
                f"  Generating error examples from Dmini (size {len(Dmini)}) for prompt: {prompt_candidate[:50]}...")
            error_examples_str = _get_error_examples_string(
                prompt_candidate,
                Dmini,
                config_obj.llm_task_model_name,
                config_obj.max_error_examples_for_gradient if hasattr(config_obj,
                                                                      'max_error_examples_for_gradient') else 5
            )

    # 2. Generate Textual Gradients (LLM_gradient)
    textual_gradients = ""
    if error_examples_str:  # Only generate gradients if there are errors to learn from
        logger.info(f"  Generating textual gradients using LLM: {config_obj.llm_gradient_model_name}...")
        try:
            textual_gradients = llm_wrappers.generate_textual_gradients(
                current_classifier_prompt=prompt_candidate,
                error_examples_str=error_examples_str,
                model_name=config.llm_gradient_model_name,
                num_gradients_to_generate=config.num_gradients_to_generate
            )
            logger.info(f"    Generated {len(textual_gradients)} gradients.")
        except Exception as e:
            logger.error(f"    Error during gradient generation: {e}")
            textual_gradients = "" # Ensure it's a list
    else:
        logger.info("  No error examples found or Dmini was empty/too small, skipping gradient generation.")

    # 3. Edit prompt using gradients (LLM_edit)
    if textual_gradients:
        for _ in range(config.beam_width):
            try:
                llm_output = llm_wrappers.edit_prompt_with_gradient(
                    original_prompt=prompt_candidate,
                    textual_gradient=textual_gradients,
                    model_name=config.llm_gradient_model_name,
                    num_edits_per_gradient=""
                )
                newly_generated_prompts.append(llm_output)
            except Exception as e:
                logger.error(f"      Error during prompt editing with gradient ': {e}")
    else:
        logger.info("  No gradients available for editing.")

    # 4. Paraphrase the original prompt candidate (LLM_paraphrase)
    # logger.info(f"  Paraphrasing original prompt candidate using LLM: {config_obj.llm_paraphrase_model_name}...")
    # try:
    #     paraphrased_prompts = llm_wrappers.paraphrase_prompt(
    #         original_prompt=prompt_candidate,
    #         model_name=config_obj.llm_paraphrase_model_name,
    #         num_paraphrases=config_obj.num_paraphrases_to_generate
    #     )
    #     logger.info(f"    Generated {len(paraphrased_prompts)} paraphrases.")
    #     newly_generated_prompts.extend(paraphrased_prompts)
    # except Exception as e:
    #     logger.error(f"    Error during paraphrasing: {e}")

    # for prompt in newly_generated_prompts:
    #
    #
    # # Remove duplicates and empty strings
    # unique_new_prompts = sorted(list(set(p for p in newly_generated_prompts if p and p.strip())))
    # logger.info(
    #     f"  Expansion of '{prompt_candidate[:50]}...' resulted in {len(unique_new_prompts)} unique new prompts.")
    return newly_generated_prompts


def expand_prompt_set(current_beam, Dtr_train_df, config_obj):
    """
    Expands each prompt in the current beam to generate a larger set of candidates.
    """
    all_new_candidate_prompts = []
    logger.info(f"Expanding {len(current_beam)} prompts in the current beam.")
    for i, p_candidate in enumerate(current_beam):
        logger.info(f"Processing prompt {i + 1}/{len(current_beam)} from beam for expansion...")
        expanded_from_p = expand_single_prompt(p_candidate, Dtr_train_df, config_obj)
        all_new_candidate_prompts.extend(expanded_from_p)

    # Add original beam prompts to ensure they are considered if no good expansions are found
    all_new_candidate_prompts.extend(current_beam)
    #
    #
    # # De-duplicate across all generated and original prompts
    # unique_total_candidates = sorted(list(set(p for p in all_new_candidate_prompts if p and p.strip())))
    # logger.info(f"Total unique candidate prompts after expansion and adding beam: {len(unique_total_candidates)}")
    return all_new_candidate_prompts


def select_prompts_fallback(candidate_prompts, Dtr_train_df, beam_width, config_obj):
    """
    Fallback selection strategy: Evaluate all candidates on a fixed sample of Dtr_train
    and pick the top `beam_width`.
    """
    if not candidate_prompts:
        logger.warning("No candidate prompts to select from.")
        return []
    if len(candidate_prompts) <= beam_width:
        logger.info(
            f"Number of candidates ({len(candidate_prompts)}) is less than or equal to beam width ({beam_width}). Returning all candidates.")
        return candidate_prompts

    logger.info(f"Selecting {beam_width} prompts from {len(candidate_prompts)} candidates using fallback evaluation.")

    if Dtr_train_df.empty or len(Dtr_train_df) < config_obj.fallback_eval_size:
        logger.warning(
            f"Training data for selection is too small ({len(Dtr_train_df)}) or empty. Cannot reliably score. Returning random subset or first N.")
        random.shuffle(candidate_prompts)
        return candidate_prompts[:beam_width]

    D_eval_sample = data_utils.sample_data_for_selection(
        Dtr_train_df,
        config_obj.fallback_eval_size
    )
    if D_eval_sample.empty:
        logger.error("Failed to sample D_eval_sample for selection. Returning random subset.")
        random.shuffle(candidate_prompts)
        return candidate_prompts[:beam_width]

    logger.info(f"Evaluating candidates on a sample of {len(D_eval_sample)} instances from Dtr_train.")

    prompt_scores = {}
    for i, p_cand in enumerate(candidate_prompts):
        logger.debug(f"  Evaluating candidate {i + 1}/{len(candidate_prompts)} for selection: {p_cand[:70]}...")
        try:
            score = metrics.calculate_metric_for_prompt(
                p_cand,
                D_eval_sample,  # Evaluate on the sampled data
                config_obj.llm_task_model_name,
                config_obj.metric_type,
                max_samples_for_eval=len(D_eval_sample)  # Ensure it uses the full sample
            )
            prompt_scores[p_cand] = score
            logger.debug(f"    Score for candidate {i + 1}: {score:.4f}")
        except Exception as e:
            logger.error(
                f"    Error calculating score for candidate '{p_cand[:50]}...': {e}. Assigning worst score (-1).")
            prompt_scores[p_cand] = -1.0  # Assign a very low score

    # Sort prompts by score in descending order
    sorted_prompts = sorted(prompt_scores.items(), key=lambda item: item[1], reverse=True)

    selected_beam = [p for p, score in sorted_prompts[:beam_width]]

    if selected_beam:
        logger.info(
            f"Selected new beam of {len(selected_beam)} prompts. Best score in selection: {prompt_scores[selected_beam[0]]:.4f}")
    else:
        logger.warning("Selection resulted in an empty beam.")
        # Fallback if all scores were terrible or errors occurred
        if candidate_prompts:
            logger.warning("Returning a random subset of original candidates as a last resort.")
            random.shuffle(candidate_prompts)
            return candidate_prompts[:beam_width]
    return selected_beam


# --- Placeholder for Advanced Selection Strategies (Successive Rejects, UCB) ---
# def select_prompts_successive_rejects(candidate_prompts, Dtr_train_df, beam_width, config_obj, total_query_budget):
#     logger.info("Using Successive Rejects selection strategy...")
#     # Implementation based on the paper's description
#     # Needs careful budget allocation (nk) per round
#     # ...
#     pass

# def select_prompts_ucb(candidate_prompts, Dtr_train_df, beam_width, config_obj, T_timesteps):
#     logger.info("Using UCB selection strategy...")
#     # Implementation of UCB1 algorithm for multi-armed bandits
#     # Each prompt is an "arm"
#     # Needs to track counts (N_i) and empirical means (mu_i) for each arm
#     # ...
#     pass
# ---------------------------------------------------------------------------------

def run_protegi(initial_prompt, Dtr_train_df, config_obj):
    """
    Main function to run the ProTeGi algorithm.

    Args:
        initial_prompt (str): The starting prompt.
        Dtr_train_df (pd.DataFrame): The training dataset.
        config_obj (module): The configuration module.

    Returns:
        str: The best prompt found after the search.
    """
    logger.info("Starting ProTeGi algorithm...")
    logger.info(f"Initial prompt: {initial_prompt[:100]}...")
    logger.info(f"Search depth: {config_obj.search_depth}, Beam width: {config_obj.beam_width}")
    logger.info(f"Task LLM: {config_obj.llm_task_model_name}, Metric: {config_obj.metric_type}")

    current_beam = [initial_prompt]
    all_time_best_prompt_overall = initial_prompt
    # Evaluate initial prompt on the full Dtr_train for a baseline
    try:
        all_time_best_score_overall = metrics.calculate_metric_for_prompt(
            initial_prompt, Dtr_train_df, config_obj.llm_task_model_name, config_obj.metric_type
        )
        logger.info(f"Initial prompt score on full Dtr_train: {all_time_best_score_overall:.4f}")
    except Exception as e:
        logger.error(f"Could not calculate initial prompt score: {e}")
        all_time_best_score_overall = -1.0

    for i in range(config_obj.search_depth):
        logger.info(f"\n--- Iteration {i + 1}/{config_obj.search_depth} ---")

        # 1. Expansion Phase
        logger.info("Phase 1: Expansion")
        all_candidate_prompts = expand_prompt_set(current_beam, Dtr_train_df, config_obj)

        if not all_candidate_prompts:
            logger.warning("Expansion phase yielded no candidate prompts. Stopping early.")
            break

        # 超过宽度进行评估 缩减
        if len(all_candidate_prompts) > config.beam_width:
            result_eval = []
            for candidate in all_candidate_prompts:
                try:
                    score = metrics.calculate_metric_for_prompt(
                        candidate,
                        Dtr_train_df,
                        config_obj.llm_task_model_name,
                        config_obj.metric_type
                    )

                except Exception as e:
                    logger.error(f"    Error calculating score for candidate  {e}. Assigning worst score (-1).")
                    score = -1.0

                result_eval.append((score, candidate))

            result_eval = sorted(result_eval, key=lambda item: item[0])[:config.beam_width]


            with open(f"results/beam_{i}.json", "w") as f:
                json.dump(result_eval, f, indent=4)

            logger.info(f"select {result_eval} for next iteration {i + 1}/{config_obj.search_depth}")

            current_beam =[item[1] for item in result_eval]

        else:
            current_beam = all_candidate_prompts

        # logger.info(f"Generated {len(all_candidate_prompts)} total candidates for selection (incl. previous beam).")
        #
        # # 2. Selection Phase
        # logger.info("Phase 2: Selection")
        # # Choose selection strategy based on config (e.g., config.selection_strategy)
        # # For now, using the fallback
        # if config_obj.selection_strategy == "fallback":
        #     current_beam = select_prompts_fallback(all_candidate_prompts, Dtr_train_df, config_obj.beam_width,
        #                                            config_obj)
        # # elif config_obj.selection_strategy == "sr":
        # #     current_beam = select_prompts_successive_rejects(all_candidate_prompts, Dtr_train_df, config_obj.beam_width, config_obj, config_obj.sr_total_budget)
        # # elif config_obj.selection_strategy == "ucb":
        # #     current_beam = select_prompts_ucb(all_candidate_prompts, Dtr_train_df, config_obj.beam_width, config_obj, config_obj.ucb_T_timesteps)
        # else:
        #     logger.warning(f"Unknown selection strategy: {config_obj.selection_strategy}. Defaulting to fallback.")
        #     current_beam = select_prompts_fallback(all_candidate_prompts, Dtr_train_df, config_obj.beam_width,
        #                                            config_obj)
        #
        # if not current_beam:
        #     logger.warning("Selection phase resulted in an empty beam. Stopping early.")
        #     # Potentially revert to previous beam or best known prompt if this happens
        #     if all_time_best_prompt_overall:
        #         current_beam = [all_time_best_prompt_overall]  # Try to recover
        #         logger.warning(f"Reverting beam to last known best prompt: {all_time_best_prompt_overall[:50]}...")
        #     else:  # Should not happen if initial prompt is set
        #         break
        #
        # # Log best prompt in current beam (based on selection sample score)
        # if current_beam:
        #     # The score used for selection was on a sample. For logging, let's re-evaluate the top one on full Dtr_train.
        #     # Or, if selection already provided scores, use that.
        #     # For simplicity, we'll just log the prompt. A more rigorous approach would re-score.
        #     logger.info(f"Iteration {i + 1} complete. Current beam (top prompt): {current_beam[0][:100]}...")
        #
        #     # Check if any prompt in the current beam is better than the all_time_best on the full training set
        #     for p_beam in current_beam:
        #         try:
        #             current_p_beam_score = metrics.calculate_metric_for_prompt(
        #                 p_beam, Dtr_train_df, config_obj.llm_task_model_name, config_obj.metric_type
        #             )
        #             logger.info(f"  Prompt in beam '{p_beam[:50]}...' score on Dtr_train: {current_p_beam_score:.4f}")
        #             if current_p_beam_score > all_time_best_score_overall:
        #                 all_time_best_score_overall = current_p_beam_score
        #                 all_time_best_prompt_overall = p_beam
        #                 logger.info(
        #                     f"  *** New best overall prompt found (Score: {all_time_best_score_overall:.4f}): {all_time_best_prompt_overall[:100]}... ***")
        #         except Exception as e:
        #             logger.error(f"  Error scoring prompt '{p_beam[:50]}...' from beam on Dtr_train: {e}")
        # else:
        #     logger.warning(f"Iteration {i + 1} complete. Beam is empty.")

    logger.info("\n--- ProTeGi Algorithm Finished ---")
    if not current_beam and not all_time_best_prompt_overall:
        logger.error("No best prompt found. Returning initial prompt.")
        return initial_prompt

    # The best prompt is tracked by all_time_best_prompt_overall based on full Dtr_train evaluation
    logger.info(
        f"Best prompt found overall with score {all_time_best_score_overall:.4f} on Dtr_train: \n{all_time_best_prompt_overall}")
    return all_time_best_prompt_overall


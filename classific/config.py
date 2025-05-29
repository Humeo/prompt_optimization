# In config.py
# ... (existing configs) ...
import os

base_url="http://localhost:11434/v1"
api_key="fake"
llm_task_model_name="qwen3:8b"
llm_gradient_model_name="qwen3:8b"
llm_paraphrase_model_name="qwen3:8b"
llm_task_max_tokens=4096
# Data paths and split
data_csv_path = "demo_data/labeled_demo_data_aitmg_202504.csv"  # IMPORTANT: Update this
train_test_split_ratio = 0.1 # 80% for training (Dtr), 20% for final hold-out test (Dte)
random_seed = 42 # For reproducible splits and sampling
max_samples_for_final_eval=5
in_ollama=True

# LLM Model Names (ensure these are valid for your chosen API provider)
# Example for OpenAI:
# llm_task_model_name = "gpt-3.5-turbo-instruct" # Or "gpt-3.5-turbo" if using chat completions endpoint
# llm_gradient_model_name = "gpt-4-turbo-preview" # Needs to be good at reasoning
# llm_edit_model_name = "gpt-4-turbo-preview"     # Needs to be good at rewriting
# llm_paraphrase_model_name = "gpt-3.5-turbo-instruct"

# Metric for evaluation
metric_type = 'f1_macro' # 'f1_weighted', 'accuracy'

# ProTeGi Algorithm Parameters
beam_width = 3
beam_expand = 2
search_depth = 5 # Number of optimization iterations
minibatch_size_for_errors = 100 # For Algorithm 2, line 1
num_gradients_to_generate = 4  # 'm' in paper's Algorithm 2, line 3 (number of g_i)
num_edits_per_gradient = 2     # 'q' in paper's Algorithm 2, line 4 (number of p'_ij)
num_paraphrases_per_edit = 1   # 'm' in paper's Algorithm 2, line 5 (paraphrases for p'_ij)

# Selection Strategy & Parameters
selection_strategy = 'fallback' # 'successive_rejects', 'ucb', or 'fallback'
# For 'successive_rejects' (if you implement the full version)
sr_query_budget_per_selection = 500 # Total LLM calls (evaluations) allowed for one Select_b step
# For 'ucb'
ucb_T_timesteps = 50 # Total pulls for UCB in one Select_b step
ucb_c_exploration = 1.414 # Exploration parameter for UCB
ucb_data_sample_size = 5 # Number of data points to evaluate a UCB arm on per pull
# For 'fallback' or general quick evaluations
fallback_eval_size = 20 # Number of samples to evaluate each candidate on if using fallback selection
max_samples_for_quick_eval = 15 # For intermediate logging of beam performance

# LLM API specific (example for OpenAI)
openai_api_key = os.getenv("OPENAI_API_KEY") # Recommended to use environment variables
# Add other provider configs if needed (e.g., Google Gemini API key, endpoint)

# Paths
initial_classifier_prompt_path = "prompts/initial_classifier_prompt.txt"
# (gradient_generation_prompt, prompt_editing_prompt, paraphrasing_prompt paths are used internally by llm_wrappers)

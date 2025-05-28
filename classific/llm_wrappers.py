# llm_wrappers.py

import os
import logging
import re

from openai import timeout
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Attempt to import LLM SDKs
try:
    import openai
except ImportError:
    openai = None
    logging.warning("OpenAI SDK not installed. OpenAI functionalities will not be available.")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logging.warning("Google Generative AI SDK not installed. Gemini functionalities will not be available.")

import config  # Your project's configuration file

logger = logging.getLogger(__name__)

# --- Meta-Prompt Loading ---
# Define paths to your meta-prompts (relative to where this script might be run from, or use absolute paths)
PROMPT_DIR = "prompts"  # Assuming a 'prompts' subdirectory
GRADIENT_GENERATION_PROMPT_PATH = os.path.join(PROMPT_DIR, "gradient_generation_prompt.txt")
PROMPT_EDITING_PROMPT_PATH = os.path.join(PROMPT_DIR, "prompt_editing_prompt.txt")
PARAPHRASING_PROMPT_PATH = os.path.join(PROMPT_DIR, "paraphrasing_prompt.txt")


# Load meta-prompts at module level
def _load_meta_prompt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Meta-prompt file not found: {path}")
        # Return a placeholder or raise an error to prevent execution with missing critical components
        raise FileNotFoundError(f"Critical meta-prompt missing: {path}")
    except Exception as e:
        logger.error(f"Error loading meta-prompt from {path}: {e}")
        raise


# These will be loaded once when the module is imported
# Ensure these files exist with the content provided in earlier discussions
META_PROMPT_GRADIENT_GENERATION = _load_meta_prompt(GRADIENT_GENERATION_PROMPT_PATH)
META_PROMPT_PROMPT_EDITING = _load_meta_prompt(PROMPT_EDITING_PROMPT_PATH)
META_PROMPT_PARAPHRASING = _load_meta_prompt(PARAPHRASING_PROMPT_PATH)


# --- LLM API Call Abstraction ---

# Define custom exceptions for LLM interactions for tenacity
class LLMAPIError(Exception): pass


class LLMRateLimitError(Exception): pass  # OpenAI specific, but good to have


class LLMOutputParsingError(Exception): pass


class LLMContentFilterError(Exception): pass  # For when content is blocked


# Configure OpenAI client if available and API key is set
# if openai and config.openai_api_key:
#     openai.api_key = config.openai_api_key
#     # For OpenAI SDK v1.0.0+
#     # openai_client = openai.OpenAI(api_key=config.openai_api_key)
# else:
#     # openai_client = None
#     if not config.openai_api_key and openai:
#         logger.warning("OpenAI SDK is imported, but OPENAI_API_KEY is not set in config.")
#
# # Configure Gemini client if available and API key is set
# if genai and config.gemini_api_key:
#     try:
#         genai.configure(api_key=config.gemini_api_key)
#     except Exception as e:
#         logger.error(f"Failed to configure Google Generative AI: {e}")
#         genai = None  # Disable genai if configuration fails
# else:
#     if not config.gemini_api_key and genai:
#         logger.warning("Google Generative AI SDK is imported, but GEMINI_API_KEY is not set in config.")


# @retry(
#     wait=wait_exponential(multiplier=1, min=2, max=30),  # Exponential backoff: 2s, 4s, 8s, 16s, 30s, 30s...
#     stop=stop_after_attempt(config.llm_max_retries if hasattr(config, 'llm_max_retries') else 5),  # Max 5 retries
#     retry=retry_if_exception_type(
#         (LLMAPIError, LLMRateLimitError, openai.error.RateLimitError if openai else LLMRateLimitError))
#     # Add more specific exceptions
# )
def _call_llm(full_prompt_text, model_name, temperature=0.7, max_tokens=512, stop_sequences=None):
    """
    Core function to call an LLM.
    Detects provider based on model_name prefix or a configured mapping.
    """
    logger.debug(
        f"Calling LLM: {model_name} with prompt (first 100 chars): {full_prompt_text[:100].replace(os.linesep, ' ')}...")

    # --- OpenAI Logic ---
    client = openai.OpenAI(
        base_url=config.base_url,
        api_key=config.api_key
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt_text}],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop_sequences,
        extra_body={"chat_template_kwargs":{"enable_thinking":False}}
    )
    content = response.choices[0].message.content.strip()
            # else:
            #     # Use Completion for models like "text-davinci-003", "gpt-3.5-turbo-instruct"
            #     response = openai.Completion.create(
            #         model=model_name,
            #         prompt=full_prompt_text,
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #         stop=stop_sequences
            #     )
            #     content = response.choices[0].text.strip()
            #
            # logger.debug(f"LLM Raw Response ({model_name}): {content[:100].replace(os.linesep, ' ')}...")
            # return content

    #     except openai.error.RateLimitError as e:
    #         logger.warning(f"OpenAI Rate Limit Error for {model_name}: {e}. Retrying...")
    #         raise LLMRateLimitError(str(e))
    #     except openai.error.AuthenticationError as e:
    #         logger.error(f"OpenAI Authentication Error: {e}. Check your API key.")
    #         raise LLMAPIError(f"OpenAI Authentication Error: {e}")
    #     except openai.error.APIError as e:
    #         logger.error(f"OpenAI API Error for {model_name}: {e}")
    #         raise LLMAPIError(str(e))
    #     except Exception as e:  # Catch any other OpenAI specific errors
    #         logger.error(f"Unexpected error with OpenAI model {model_name}: {e}")
    #         raise LLMAPIError(f"Unexpected OpenAI error: {e}")
    #
    # # --- Google Gemini Logic ---
    # elif model_name.startswith("gemini-"):  # Google Gemini models
    #     if not genai or not config.gemini_api_key:
    #         logger.error(f"Google Generative AI SDK not available or API key not set for model: {model_name}")
    #         raise LLMAPIError("Gemini not configured.")
    #     try:
    #         gemini_model = genai.GenerativeModel(
    #             model_name,
    #             generation_config=genai.types.GenerationConfig(  # Use types for safety
    #                 temperature=temperature,
    #                 max_output_tokens=max_tokens,
    #                 # stop_sequences=stop_sequences # Gemini API might handle stop differently or not via config here
    #             )
    #             # safety_settings=... # Add safety settings from config if needed
    #         )
    #         # Gemini typically uses chat even for single turns with newer models
    #         # For simple text generation, sending content directly might work for some models.
    #         # Using start_chat for more consistent behavior with Gemini Pro models.
    #         chat = gemini_model.start_chat(history=[])  # Fresh chat for each call
    #         response = chat.send_message(full_prompt_text)
    #
    #         if not response.parts:
    #             logger.warning(
    #                 f"Gemini response for model {model_name} has no parts. Prompt: '{full_prompt_text[:100]}...'")
    #             if response.prompt_feedback and response.prompt_feedback.block_reason:
    #                 logger.error(f"Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason_message}")
    #                 raise LLMContentFilterError(
    #                     f"Gemini prompt blocked: {response.prompt_feedback.block_reason_message}")
    #             return ""  # Or raise an error indicating no valid response
    #
    #         content = response.text.strip()  # .text usually concatenates parts
    #         logger.debug(f"LLM Raw Response ({model_name}): {content[:100].replace(os.linesep, ' ')}...")
    return content

    #     except Exception as e:
    #         logger.error(f"Error calling Gemini model {model_name}: {e}")
    #         # Check for specific Gemini exceptions if available in SDK and raise appropriately
    #         raise LLMAPIError(f"Gemini API error: {e}")
    #
    # else:
    #     logger.error(f"Unsupported LLM model/provider for: {model_name}")
    #     raise ValueError(f"Unsupported LLM model/provider: {model_name}")


# --- Specific Task Wrappers ---

def classify_instance(classifier_prompt_text, req_text, rsp_text, model_name):
    """
    Classifies a given request/response pair using the provided classifier prompt.
    Expected output: "SUCCESS", "FAILURE", or "UNKNOWN".
    """
    # Construct the full prompt for the task LLM
    # This format should match what your initial_classifier_prompt expects
    # and how it defines the placeholders for request and response.
    # Example:
    # Your initial_classifier_prompt.txt might end with:
    # ---
    # Request:
    # {request_data}
    # ---
    # Response:
    # {response_data}
    # ---
    # Based on the rules above, classify this interaction.
    # Your output should be ONLY one of these three words: SUCCESS, FAILURE, or UNKNOWN.
    # Label:
    #
    # So, we need to append the actual req/rsp data here.
    # A more robust way is to ensure the classifier_prompt_text itself has placeholders.
    # For now, let's assume classifier_prompt_text is the *full* template including instructions
    # and we just need to fill in the data.

    # Let's assume the classifier_prompt_text is a template with {request_data} and {response_data}
    try:
        full_task_prompt = classifier_prompt_text.format(req_data=req_text, rsp_data=rsp_text)
    except KeyError:
        # Fallback if .format fails: simple concatenation (less ideal)
        # This assumes classifier_prompt_text ends in a way that req/rsp can be appended.
        logger.warning(
            "Classifier prompt does not seem to have {request_data} and {response_data} placeholders. Using simple concatenation.")
        full_task_prompt = f"{classifier_prompt_text}\n\nRequest:\n{req_text}\n\nResponse:\n{rsp_text}\n\nLabel:"

    if config.in_ollama:
        full_task_prompt += "\n\n/no_think"

    raw_output = _call_llm(
        full_task_prompt,
        model_name,
        temperature=config.llm_task_temperature if hasattr(config, 'llm_task_temperature') else 0.0,
        # Low temp for classification
        max_tokens=config.llm_task_max_tokens if hasattr(config, 'llm_task_max_tokens') else 10
    )

    # Parse the output
    output_upper = raw_output.strip().upper()
    if "SUCCESS" in output_upper:  # Be a bit lenient with parsing
        return "SUCCESS"
    elif "FAILURE" in output_upper:
        return "FAILURE"
    elif "UNKNOWN" in output_upper:
        return "UNKNOWN"
    else:
        logger.warning(
            f"Classifier LLM ({model_name}) returned unexpected output: '{raw_output}'. Defaulting to UNKNOWN.")
        return "UNKNOWN"  # Or raise LLMOutputParsingError


def generate_textual_gradients(current_classifier_prompt, error_examples_str, model_name, num_gradients_to_generate):
    """
    Generates textual gradients (suggestions for improvement) for a classifier prompt.
    """
    full_gradient_prompt = META_PROMPT_GRADIENT_GENERATION.format(
        current_classification_prompt=current_classifier_prompt,
        error_examples_string=error_examples_str,
        num_gradients=num_gradients_to_generate
    )
    raw_output = _call_llm(
        full_gradient_prompt,
        model_name,
        temperature=config.llm_gradient_temperature if hasattr(config, 'llm_gradient_temperature') else 0.7,
        max_tokens=config.llm_task_max_tokens
    )

    # Parse the output - assuming gradients are separated by a specific string or numbered
    # The meta-prompt should ask for a specific separator like "---GRADIENT_SEPARATOR---"
    # or numbered list "1. Gradient one...\n2. Gradient two..."
    gradients = []
    # Option 1: Custom separator
    # separator = "---GRADIENT_SEPARATOR---"  # Make this consistent with your meta-prompt
    # if separator in raw_output:
    #     gradients = [g.strip() for g in raw_output.split(separator) if g.strip()]
    # else:
    #     # Option 2: Numbered list (e.g., "1. ...", "2. ...")
    #     # This regex tries to find numbered items. It's not perfect.
    #     found_gradients = re.findall(r"^\s*\d+\.\s*(.+)", raw_output, re.MULTILINE)
    #     if found_gradients:
    #         gradients = [g.strip() for g in found_gradients if g.strip()]
    #     else:
    #         # Fallback: if no clear separation, assume the whole output is one gradient, or split by double newline
    #         logger.warning(
    #             f"Could not parse multiple gradients from LLM_gradient output. Using fallback splitting. Output: {raw_output[:200]}")
    #         gradients = [g.strip() for g in raw_output.split("\n\n") if g.strip()]  # Common fallback
    #
    # if not gradients and raw_output.strip():  # If still no gradients but there's output
    #     gradients = [raw_output.strip()]

    logger.info(f"Generated {len(gradients)} raw textual gradients.")
    return raw_output  # Return up to the number requested


def edit_prompt_with_gradient(original_prompt, textual_gradient, model_name, num_edits_per_gradient):
    """
    Edits an original prompt based on a textual gradient.
    """
    full_edit_prompt = META_PROMPT_PROMPT_EDITING.format(
        current_classification_prompt=original_prompt,
        textual_gradient=textual_gradient,
    )

    if config.in_ollama:
        full_edit_prompt += "\n\n/no_think"

    raw_output = _call_llm(
        full_edit_prompt,
        model_name,
        temperature=1,
        max_tokens=config.llm_edit_max_tokens if hasattr(config, 'llm_edit_max_tokens') else 4096  # Prompts can be long
    )

    cleaned_text_multiline = re.sub(r"<think>(.*?)</think>", "", raw_output, flags=re.DOTALL).strip()
    # # Parse output - similar to gradients, expect a separator or numbering
    # edited_prompts = []
    # separator = "---EDIT_SEPARATOR---"  # Make this consistent with your meta-prompt
    # if separator in raw_output:
    #     edited_prompts = [p.strip() for p in raw_output.split(separator) if p.strip()]
    # else:
    #     found_edits = re.findall(
    #         r"^\s*Improved Prompt Version \d+:\s*([\s\S]*?)(?=\n\s*Improved Prompt Version \d+:|\Z)", raw_output,
    #         re.MULTILINE)
    #     if found_edits:
    #         edited_prompts = [p.strip() for p in found_edits if p.strip()]
    #     else:
    #         logger.warning(
    #             f"Could not parse multiple edited prompts from LLM_edit output. Using fallback splitting. Output: {raw_output[:200]}")
    #         edited_prompts = [p.strip() for p in raw_output.split("\n\n---\n\n") if
    #                           p.strip()]  # Paper mentions this separator
    #
    # if not edited_prompts and raw_output.strip():
    #     edited_prompts = [raw_output.strip()]
    #
    # logger.info(f"Generated {len(edited_prompts)} raw edited prompts from one gradient.")
    return cleaned_text_multiline


def paraphrase_prompt(original_prompt, model_name, num_paraphrases):
    """
    Generates paraphrased versions of an original prompt.
    """
    full_paraphrase_prompt = META_PROMPT_PARAPHRASING.format(
        original_prompt=original_prompt,
        num_versions=num_paraphrases
    )
    raw_output = _call_llm(
        full_paraphrase_prompt,
        model_name,
        temperature=config.llm_paraphrase_temperature if hasattr(config, 'llm_paraphrase_temperature') else 0.7,
        max_tokens=config.llm_paraphrase_max_tokens if hasattr(config, 'llm_paraphrase_max_tokens') else 2048
        # Prompts can be long
    )

    # Parse output
    paraphrased_prompts = []
    separator = "---PARAPHRASE_SEPARATOR---"  # Make this consistent with your meta-prompt
    if separator in raw_output:
        paraphrased_prompts = [p.strip() for p in raw_output.split(separator) if p.strip()]
    else:
        # Fallback similar to edits
        found_paraphrases = re.findall(r"^\s*Paraphrased Version \d+:\s*([\s\S]*?)(?=\n\s*Paraphrased Version \d+:|\Z)",
                                       raw_output, re.MULTILINE)
        if found_paraphrases:
            paraphrased_prompts = [p.strip() for p in found_paraphrases if p.strip()]
        else:
            logger.warning(
                f"Could not parse multiple paraphrased prompts from LLM_paraphrase output. Using fallback splitting. Output: {raw_output[:200]}")
            paraphrased_prompts = [p.strip() for p in raw_output.split("\n\n---\n\n") if p.strip()]

    if not paraphrased_prompts and raw_output.strip():
        paraphrased_prompts = [raw_output.strip()]

    logger.info(f"Generated {len(paraphrased_prompts)} raw paraphrased prompts.")
    return paraphrased_prompts[:num_paraphrases]


# --- Example Usage (for testing this module directly) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(message)s")
    logger.info("Testing llm_wrappers.py...")

    # Ensure config.py has API keys and model names set
    # Example:
    # config.openai_api_key = "sk-..." or os.getenv("OPENAI_API_KEY")
    # config.llm_task_model_name = "gpt-3.5-turbo-instruct" (or a chat model like "gpt-3.5-turbo")
    # config.llm_gradient_model_name = "gpt-4-turbo-preview"
    # config.llm_edit_model_name = "gpt-4-turbo-preview"
    # config.llm_paraphrase_model_name = "gpt-3.5-turbo-instruct"

    # Create dummy meta-prompt files in a 'prompts' subdirectory for this test to run
    os.makedirs(PROMPT_DIR, exist_ok=True)
    if not os.path.exists(GRADIENT_GENERATION_PROMPT_PATH):
        with open(GRADIENT_GENERATION_PROMPT_PATH, "w") as f:
            f.write(
                "Current prompt:\n{current_prompt}\n\nError examples:\n{error_examples}\n\nSuggest {num_suggestions} ways to improve the prompt to fix these errors. Separate suggestions with '---GRADIENT_SEPARATOR---'.")
    if not os.path.exists(PROMPT_EDITING_PROMPT_PATH):
        with open(PROMPT_EDITING_PROMPT_PATH, "w") as f:
            f.write(
                "Original prompt:\n{original_prompt}\n\nImprovement suggestion:\n{textual_gradient}\n\nRewrite the original prompt in {num_versions} ways based on the suggestion. Separate versions with '---EDIT_SEPARATOR---'.")
    if not os.path.exists(PARAPHRASING_PROMPT_PATH):
        with open(PARAPHRASING_PROMPT_PATH, "w") as f:
            f.write(
                "Original prompt:\n{original_prompt}\n\nParaphrase this prompt in {num_versions} ways while keeping the core meaning. Separate versions with '---PARAPHRASE_SEPARATOR---'.")

    # Reload meta-prompts if they were just created
    META_PROMPT_GRADIENT_GENERATION = _load_meta_prompt(GRADIENT_GENERATION_PROMPT_PATH)
    META_PROMPT_PROMPT_EDITING = _load_meta_prompt(PROMPT_EDITING_PROMPT_PATH)
    META_PROMPT_PARAPHRASING = _load_meta_prompt(PARAPHRASING_PROMPT_PATH)

    # Test 1: Classification
    logger.info("\n--- Testing classify_instance ---")
    # This initial_classifier_prompt should be your actual detailed one
    # For this test, a simplified version that expects .format()
    test_classifier_prompt = """
    You are an HTTP interaction classifier.
    Classify the following interaction as SUCCESS, FAILURE, or UNKNOWN.
    SUCCESS: The user's request was understood and a valid, expected positive outcome was achieved.
    FAILURE: The user's request led to an error, was denied, or a negative outcome occurred.
    UNKNOWN: The interaction is ambiguous, a generic acknowledgement, or it's unclear if the core request succeeded or failed.

    Request:
    {request_data}
    ---
    Response:
    {response_data}
    ---
    Label:"""
    req_example = "GET /api/data HTTP/1.1\nHost: example.com"
    rsp_example_success = "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"status\": \"success\", \"data\": [1,2,3]}"
    rsp_example_failure = "HTTP/1.1 500 Internal Server Error\n\nError processing request."

    if config.llm_task_model_name:
        try:
            label1 = classify_instance(test_classifier_prompt, req_example, rsp_example_success,
                                       config.llm_task_model_name)
            logger.info(f"Classification for success example: {label1}")
            label2 = classify_instance(test_classifier_prompt, req_example, rsp_example_failure,
                                       config.llm_task_model_name)
            logger.info(f"Classification for success example: {label2}")
        except Exception as e:
            logger.error(f"Failed to classify example: {e}")
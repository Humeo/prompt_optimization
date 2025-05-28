# In data_utils.py
from sklearn.model_selection import train_test_split as sklearn_split # for stratified split
import config
from main import logger
import pandas as pd
import logging
from sklearn.model_selection import train_test_split as sklearn_split  # For stratified sampling/splitting
import config  # To access config.random_seed

logger = logging.getLogger(__name__)


def load_data(csv_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        # Basic validation: Check for required columns
        required_columns = ['uuid', 'req', 'rsp', 'label']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"CSV file missing required columns: {missing}")
            raise ValueError(f"CSV file missing required columns: {missing}")
        logger.info(f"Data loaded successfully from {csv_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at path: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {csv_path}: {e}")
        raise


def split_data(df, train_ratio=0.8, stratify_col='label'):
    """Splits data into training and test sets, optionally stratified."""
    if not (0 < train_ratio < 1):
        logger.error(f"train_ratio must be between 0 and 1, got {train_ratio}")
        raise ValueError("train_ratio must be between 0 and 1")

    if stratify_col and stratify_col in df.columns and df[stratify_col].nunique() >= 2:
        # Check if stratification is feasible (at least 2 samples per class for splitting)
        class_counts = df[stratify_col].value_counts()
        if (class_counts < 2).any() and len(df) * (1 - train_ratio) >= 1 and len(
                df) * train_ratio >= 1:  # Check if any class has <2 samples
            logger.warning(
                f"Stratification by '{stratify_col}' might be unstable due to classes with < 2 samples. Proceeding, but monitor splits.")

        try:
            train_df, test_df = sklearn_split(
                df,
                train_size=train_ratio,
                stratify=df[stratify_col],
                random_state=config.random_seed
            )
            logger.info(
                f"Data split with stratification on '{stratify_col}'. Train size: {len(train_df)}, Test size: {len(test_df)}")
            return train_df, test_df
        except ValueError as e:
            logger.warning(f"Stratified split failed ('{e}'). Falling back to non-stratified split.")
            # Fall through to non-stratified split

    # Non-stratified split
    train_df, test_df = sklearn_split(
        df,
        train_size=train_ratio,
        random_state=config.random_seed
    )
    logger.info(f"Data split without stratification. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return train_df, test_df


def sample_data_for_selection(dataset_df, sample_size, replace=False):
    """
    Samples data from the dataset for evaluating prompts during selection.

    Args:
        dataset_df (pd.DataFrame): The DataFrame to sample from (typically Dtr_train).
        sample_size (int): The desired number of samples.
        replace (bool): Whether to sample with replacement.
                        Generally False, unless sample_size > len(dataset_df).

    Returns:
        pd.DataFrame: A DataFrame containing the sampled data.
    """
    if not isinstance(dataset_df, pd.DataFrame):
        logger.error("dataset_df must be a pandas DataFrame.")
        raise TypeError("dataset_df must be a pandas DataFrame.")
    if dataset_df.empty:
        logger.warning("Attempted to sample from an empty DataFrame. Returning empty DataFrame.")
        return pd.DataFrame()
    if sample_size <= 0:
        logger.warning(f"sample_size must be positive, got {sample_size}. Returning empty DataFrame.")
        return pd.DataFrame()

    actual_sample_size = int(sample_size)

    if actual_sample_size > len(dataset_df) and not replace:
        logger.warning(
            f"Requested sample_size ({actual_sample_size}) is larger than dataset size ({len(dataset_df)}) "
            f"and replace=False. Returning the entire dataset instead."
        )
        return dataset_df.copy()  # Return a copy to avoid modifying the original if it's used elsewhere

    if actual_sample_size > len(dataset_df) and replace:
        logger.info(
            f"Requested sample_size ({actual_sample_size}) is larger than dataset size ({len(dataset_df)}). "
            f"Sampling with replacement."
        )
        return dataset_df.sample(n=actual_sample_size, replace=True, random_state=config.random_seed)

    # Standard case: sample_size <= len(dataset_df)
    return dataset_df.sample(n=actual_sample_size, replace=replace, random_state=config.random_seed)


def sample_minibatch(dataset_df, minibatch_size, stratify_col='label', replace=False):
    """
    Samples a minibatch from the dataset, typically for error collection.
    Can perform stratified sampling if specified and feasible.

    Args:
        dataset_df (pd.DataFrame): The DataFrame to sample from (typically Dtr_train).
        minibatch_size (int): The desired number of samples in the minibatch.
        stratify_col (str, optional): Column to stratify on. Defaults to 'label'.
                                      Set to None to disable stratification.
        replace (bool): Whether to sample with replacement. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled minibatch.
    """
    if not isinstance(dataset_df, pd.DataFrame):
        logger.error("dataset_df must be a pandas DataFrame.")
        raise TypeError("dataset_df must be a pandas DataFrame.")
    if dataset_df.empty:
        logger.warning("Attempted to sample minibatch from an empty DataFrame. Returning empty DataFrame.")
        return pd.DataFrame()
    if minibatch_size <= 0:
        logger.warning(f"minibatch_size must be positive, got {minibatch_size}. Returning empty DataFrame.")
        return pd.DataFrame()

    actual_minibatch_size = int(minibatch_size)

    if actual_minibatch_size > len(dataset_df) and not replace:
        logger.warning(
            f"Requested minibatch_size ({actual_minibatch_size}) is larger than dataset size ({len(dataset_df)}) "
            f"and replace=False. Returning the entire dataset as minibatch."
        )
        return dataset_df.copy()

    # Attempt stratified sampling if stratify_col is provided and valid
    perform_stratification = (
            stratify_col and
            stratify_col in dataset_df.columns and
            dataset_df[stratify_col].nunique() >= 2  # At least two classes needed for stratification
    )

    if perform_stratification:
        try:
            # sklearn's train_test_split can be used for stratified sampling by taking one part of the split.
            # This requires that each class has at least `n_splits` (which is 2 by default for train/test)
            # or for train_size/test_size to be valid.
            # A more direct way for sampling is df.groupby(stratify_col).sample(n=..., replace=...)
            # but ensuring exact total size with groupby().sample() can be tricky if n per group is small.

            # Let's use groupby().sample() for more direct control over minibatch size.
            # Calculate n_per_group, ensuring it's at least 1 if possible.
            n_groups = dataset_df[stratify_col].nunique()
            n_per_group_ideal = max(1, actual_minibatch_size // n_groups)

            # Adjust if ideal sampling leads to too few or too many samples due to rounding
            # This is a common challenge with stratified sampling to an exact total N.
            # For simplicity, we'll sample n_per_group_ideal from each group and then, if needed,
            # randomly sample more to reach actual_minibatch_size or trim if over.
            # This is a heuristic.

            sampled_df = dataset_df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), n_per_group_ideal),  # Sample at most the group size or ideal
                    replace=replace or (n_per_group_ideal > len(x)),  # Allow replacement if needed for small groups
                    random_state=config.random_seed
                )
            )

            # If we don't have enough samples, top up randomly (non-stratified)
            if len(sampled_df) < actual_minibatch_size:
                remaining_needed = actual_minibatch_size - len(sampled_df)
                if not dataset_df.index.difference(sampled_df.index).empty:  # if there are unsampled items
                    additional_samples = dataset_df.drop(sampled_df.index).sample(
                        n=min(remaining_needed, len(dataset_df.index.difference(sampled_df.index))),
                        replace=replace,  # or False if we only pick from remaining unique
                        random_state=config.random_seed
                    )
                    sampled_df = pd.concat([sampled_df, additional_samples])

            # If we have too many (unlikely with min(len(x), n_per_group_ideal) but possible if logic changes)
            if len(sampled_df) > actual_minibatch_size:
                sampled_df = sampled_df.sample(n=actual_minibatch_size, replace=False, random_state=config.random_seed)

            logger.debug(
                f"Sampled minibatch (size {len(sampled_df)}) with attempted stratification on '{stratify_col}'.")
            return sampled_df

        except Exception as e:  # Broad exception catch for complex sampling logic
            logger.warning(f"Stratified minibatch sampling failed ('{e}'). Falling back to random sampling.")
            # Fall through to random sampling

    # Default/Fallback: Random sampling
    logger.debug(
        f"Sampling minibatch (size {actual_minibatch_size}) randomly (no stratification or failed stratification).")
    return dataset_df.sample(n=actual_minibatch_size, replace=replace, random_state=config.random_seed)


def sample_minibatch_stratified(df_train, size, stratify_col='label'):
    if size >= len(df_train):
        return df_train
    if stratify_col not in df_train.columns or df_train[stratify_col].nunique() < 2:
        # Fallback to random sampling if stratification is not possible
        return df_train.sample(n=size, random_state=config.random_seed)

    # Ensure enough samples per class for stratification, otherwise sklearn might error
    # This is a simplified check; proper handling might require grouping small classes
    min_class_count = df_train[stratify_col].value_counts().min()
    if min_class_count < 2 and size > min_class_count : # A very rough check
         # If any class has only 1 sample, stratification for train_test_split might fail for small sizes
         # For sampling a minibatch, if size is small, random might be safer or use group-by apply sample
         pass # For now, let sklearn handle it, or add more robust logic

    try:
        # We want a sample, not a split, so we use train_test_split with test_size being the remainder
        # and take the 'train' part which will be our sample.
        # This is a bit of a hack for stratified sampling. A more direct way might be df.groupby().sample()
        if size / len(df_train) < 1.0:
            sample_df, _ = sklearn_split(
                df_train,
                train_size=size, # or test_size = 1 - (size / len(df_train))
                stratify=df_train[stratify_col],
                random_state=config.random_seed
            )
            return sample_df
        else: # size is >= len(df_train)
            return df_train
    except ValueError as e:
        logger.warning(f"Stratified sampling failed: {e}. Falling back to random sampling.")
        return df_train.sample(n=size, random_state=config.random_seed)

# Then in protegi_algorithm.expand_prompt:
# Dmini = data_utils.sample_minibatch_stratified(Dtr_train_df, config.minibatch_size_for_errors)

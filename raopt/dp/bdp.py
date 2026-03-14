#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz (Adapted for RAoPT BDP extension)
# ------------------------------------------------------------------------------
"""
Contains Bayesian Differential Privacy (BDP) mechanisms adapted from 
https://github.com/lange-martin/privacy-utility-bdp/
"""
import logging
from typing import Union, Iterable, Callable

import numpy as np
import pandas as pd

from raopt.utils.helpers import get_latlon_arrays, set_latlon

log = logging.getLogger()

def laplace_mechanism(input_data: np.ndarray, scale: float) -> np.ndarray:
    """
    Base Laplace mechanism to add noise to an array of inputs.
    """
    noise = np.random.laplace(0, scale, len(input_data))
    return input_data + noise

def count_active_bdp_markov_chain_bound(
        datasets: np.ndarray, epsilon: float, trans_probs: np.ndarray) -> np.ndarray:
    """
    Implements the BDP Markov Chain Bound for Laplace noise.
    :param datasets: 1D array of features that represents the sequence of states.
    :param epsilon: Privacy budget
    :param trans_probs: The measured transition probabilities of the binary Markov chain
    """
    if len(datasets.shape) == 1:
        # In RAoPT we might be passing one trajectory at a time, e.g., shape (T, )
        pass
    
    # BDP Markov Chain Bound Logic
    min_prob = np.min(trans_probs)
    max_prob = np.max(trans_probs)
    
    if min_prob <= 0:
        log.warning("Transition probability is 0, setting to a small epsilon.")
        min_prob = 1e-9

    min_eps = 4 * np.log(max_prob / min_prob)
    log.info(f"Minimum epsilon for Markov chain bound: {min_eps}")

    dp_eps = epsilon - min_eps
    if dp_eps > 0:
        return laplace_mechanism(datasets, 1.0 / dp_eps)
    else:
        raise ValueError(f"Epsilon ({epsilon}) must be greater than the minimum epsilon ({min_eps}) derived from the transition probabilities.")

def execute_generic_mechanism(df: pd.DataFrame, mechanism: Callable,
                              args: Iterable = (), kwargs: dict = {}) -> pd.DataFrame:
    """
    Execute a generic mechanism on 1D/Tabular DataFrame that bypassed spatial conversion.
    The values are expected to be stored in the 'latitude' column for compatibility with
    the DataFrame structure expected by RAoPT, or 'feature_val' if we prefer. 
    Here, we assume 'latitude' holds the generic feature to be noised.
    """
    df = df.copy()
    
    # Depending on how we structured generic_data.py, 'feature_val' or 'latitude' holds our data.
    # Let's extract 'latitude' because the standard RAoPT dictify/csv loaders rely on it.
    feature_data = df['latitude'].values
    
    # Mechanism execution
    noised_data = mechanism(feature_data, *args, **kwargs)
    
    # Store back
    df['latitude'] = noised_data
    # We also update 'feature_val' if it exists just to be consistent
    if 'feature_val' in df.columns:
        df['feature_val'] = noised_data
        
    return df

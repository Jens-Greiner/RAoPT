# RAoPT: 1D Data Preprocessing, BDP, and Training Documentation

This document provides a comprehensive overview of how 1D/generic tabular data is handled within the RAoPT framework, including preprocessing, the application of Bayesian Differential Privacy (BDP) mechanisms, and the training of the reconstruction model.

---

## 1. 1D Data Preprocessing

The preprocessing of 1D data (e.g., activity steps, heart rate) is primarily handled in [generic_data.py](file:///Users/jensgreiner/RAoPT/raopt/preprocessing/generic_data.py).

### Method: `extract_generic_data`
This function converts raw CSV files into a standardized format compatible with the RAoPT pipeline.

- **Feature Column**: The user specifies a `feature_col` (e.g., 'steps').
- **Spatial Spoofing**: Since the core RAoPT pipeline is designed for spatial trajectories, 1D data is "spoofed" for compatibility:
  - The primary feature value is stored in the `latitude` column.
  - The `longitude` column is set to `0.0`.
  - An additional `feature_val` column is created to store the raw value explicitly.
- **Trajectory Splitting**: If a `date` column is present, the data is grouped by date, creating individual "trajectories" for each day. Otherwise, the entire file is treated as a single trajectory.
- **Standard Columns**: Each record includes `trajectory_id`, `uid`, `feature_val`, `latitude`, and `longitude`.

---

## 2. Bayesian Differential Privacy (BDP)

BDP mechanisms are implemented in [bdp.py](file:///Users/jensgreiner/RAoPT/raopt/dp/bdp.py), adapted for sequential data dependencies.

### Method: `count_active_bdp_markov_chain_bound`
This is a specialized BDP mechanism for sequence data where transitions follow a Markov process.

- **Markov Chain Bound**: It computes a privacy bound based on the transition probabilities of the data.
- **Epsilon Derivation**:
  - `min_eps = 4 * log(max_prob / min_prob)` where `max_prob` and `min_prob` are transition probabilities.
  - The effective Laplace noise scale is calculated using `1 / (epsilon - min_eps)`.
- **Constraint**: The provided `epsilon` must be greater than `min_eps`.

### Method: `execute_generic_mechanism`
This is a wrapper function used to apply any BDP mechanism to a DataFrame that has bypassed spatial conversion. It extracts the data from the `latitude` column, applies the noising mechanism, and updates both `latitude` and `feature_val` with the result.

---

## 3. 1D Data Training

Training for 1D data involves specific adjustments to the `AttackModel` to ensure the LSTM treats the 1D sequence correctly.

### Model Configuration in [model.py](file:///Users/jensgreiner/RAoPT/raopt/ml/model.py):
- **`one_d_mode`**: When initializing `AttackModel`, setting `one_d_mode=True` triggers several changes:
  - **Loss Function**: Switch from spatial `euclidean_loss` to Mean Absolute Error (`mae`).
  - **Output Layer**: The model is adjusted to output only 1 dimension (latitude/feature) instead of 2 (lat/lon).
  - **Scaling**: Only the scale factor for the first dimension is applied.

### Encoding and Preprocessing:
- **Encoding**: In [encoder.py](file:///Users/jensgreiner/RAoPT/raopt/ml/encoder.py), `encode_trajectory` detects the absence of spatial columns and uses `feature_val` as the primary feature for the input matrix.
- **Reference Point**: A 1D reference point is still computed and subtracted during preprocessing to normalize the data.
- **Padding**: Like spatial trajectories, 1D sequences are padded to a fixed length using Keras' `pad_sequences`.

---

## 4. Integration in Evaluation

The [main.py](file:///Users/jensgreiner/RAoPT/raopt/eval/main.py) script automatically detects if a dataset is 1D based on the string "generic" in the dataset name.

- **`is_1d` Detection**: `is_1d = 'generic' in case['Dataset Train'].lower()`
- **Metric Selection**: In `compute_distances`, the `use_haversine` flag is set to `False` if `is_1d` is True, ensuring that simple Euclidean distances (appropriate for 1D) are used instead of spherical geometry calculations.

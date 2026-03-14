#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Author: Erik Buchholz (Adapted for RAoPT BDP extension)
# ------------------------------------------------------------------------------
"""
Pre-processes generic 1D/Tabular datasets (e.g., Activity data) into the required format.
"""
import logging
from pathlib import Path
import os
import argparse

import pandas as pd
from tqdm import tqdm

from raopt.utils import logger
from raopt.utils.config import Config

log = logging.getLogger()

def extract_generic_data(file_paths, output_file, feature_col="steps"):
    """
    Extract generic data (e.g., a time series from a CSV) and convert it into the 
    standardized RAoPT trajectory format. We will spoof coordinates by placing 
    the 1D value in the 'latitude' column, and 0 in 'longitude' for compatibility,
    or we can modify the pipeline to accept generic columns. 
    For minimal disruption to existing spatial schemas, we map the primary feature
    to 'latitude', but keep record of it being 1D data.
    
    Format output: trajectory_id, uid, latitude, longitude, timestamp (if available)
    """
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    all_data = []
    traj_id_counter = 0
    
    # Normally, activity data is one long file. We might need to split it into "trajectories" 
    # e.g., by day, or by user, etc.
    for file_path in file_paths:
        log.info(f"Processing {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # Example for simple Activity data (which might have 'date', 'steps', 'interval')
            if feature_col not in df.columns:
                log.warning(f"Feature column '{feature_col}' not found in {file_path}. Skipping.")
                continue
                
            # Fill NAs
            df[feature_col] = df[feature_col].fillna(0)
            
            # Assuming we want to split by "date" to create trajectories:
            if 'date' in df.columns:
                grouped = df.groupby('date')
                for date, group in tqdm(grouped, desc=f"Splitting by date"):
                    group = group.copy()
                    
                    # Create standard columns
                    group['trajectory_id'] = traj_id_counter
                    group['uid'] = 0
                    
                    # Store 1D feature in latitude for compatibility with some spatial parts,
                    # or explicitly keep it isolated. We will use 'feature_val'
                    group['feature_val'] = group[feature_col]
                    
                    # Fake spatial columns to prevent spatial preprocessors from crashing entirely, 
                    # though our custom executor will ignore them.
                    group['latitude'] = group[feature_col] 
                    group['longitude'] = 0.0
                    
                    all_data.append(group[['trajectory_id', 'uid', 'feature_val', 'latitude', 'longitude']])
                    traj_id_counter += 1
            else:
                # Treat entire file as one trajectory
                df['trajectory_id'] = traj_id_counter
                df['uid'] = 0
                df['feature_val'] = df[feature_col]
                df['latitude'] = df[feature_col]
                df['longitude'] = 0.0
                all_data.append(df[['trajectory_id', 'uid', 'feature_val', 'latitude', 'longitude']])
                traj_id_counter += 1
                
        except Exception as e:
            log.error(f"Error processing {file_path}: {e}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        final_df.to_csv(output_file, index=False)
        log.info(f"Saved {len(final_df)} records across {traj_id_counter} generic trajectories to {output_file}.")
    else:
        log.warning("No data extracted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess generic 1D datasets.")
    parser.add_argument('input_files', nargs='+', help="Input CSV files")
    parser.add_argument('--output', default='processed_csv/generic/originals.csv', help="Output CSV file path")
    parser.add_argument('--feature', default='steps', help="The primary feature column to extract")
    args = parser.parse_args()
    
    logger.configure_root_loger(logging.INFO, Config.get_logdir() + "generic_preprocessing.log")
    log.info("Starting generic data preprocessing...")
    extract_generic_data(args.input_files, args.output, feature_col=args.feature)

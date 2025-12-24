'''

Generate continuous time for fm
change the data path and output path with obs window

'''

import duckdb
import numpy as np
import os
import tqdm

def extract_continuous_time_industrial(raw_dir, cohort_path, output_path, max_len=164):
    """
    Industrial-grade extractor for MIMIC-IV time dynamics using DuckDB and Memmap.
    Designed for billion-scale row processing.
    """
    con = duckdb.connect(database=':memory:')
    print("Initializing DuckDB ...")

    con.execute(f"CREATE TABLE cohort AS SELECT subject_id FROM read_csv_auto('{cohort_path}') ORDER BY subject_id")
    result = con.execute("SELECT subject_id FROM cohort").fetchall()
    subject_list = [row[0] for row in result]
    num_patients = len(subject_list)
    print(f"Target Cohort Size: {num_patients} patients")

    event_tables = [
        {"file": "hosp/labevents.csv", "ts_col": "charttime"},
        {"file": "hosp/prescriptions.csv", "ts_col": "starttime"},
        {"file": "icu/inputevents.csv", "ts_col": "starttime"}
    ]
    
    queries = []
    for table in event_tables:
        full_path = os.path.join(raw_dir, table['file'])
        if os.path.exists(full_path):
            queries.append(f"SELECT subject_id, CAST({table['ts_col']} AS TIMESTAMP) as ts FROM read_csv_auto('{full_path}')")
    
    union_sql = " UNION ALL ".join(queries)
    con.execute(f"CREATE TABLE all_events AS {union_sql}")
    print("Multi-table event alignment complete.")

    res_df = con.execute("""
        WITH sorted_logs AS (
            SELECT subject_id, ts,
                   LAG(ts) OVER (PARTITION BY subject_id ORDER BY ts) as prev_ts
            FROM all_events
            WHERE subject_id IN (SELECT subject_id FROM cohort)
        )
        SELECT subject_id, 
               COALESCE(epoch(ts) - epoch(prev_ts), 0) as delta_t
        FROM sorted_logs
    """).df()
    print("Calculated precise Delta-T intervals (Seconds)...")

    print("Performing Log-Normalization for Flow Matching compatibility...")
    res_df['dt_log'] = np.log1p(res_df['delta_t'].clip(lower=0))
    t_min, t_max = res_df['dt_log'].min(), res_df['dt_log'].max()
    res_df['time_feat'] = 2 * (res_df['dt_log'] - t_min) / (t_max - t_min + 1e-8) - 1

    final_shape = (num_patients, max_len, 1)
    fp = np.memmap(output_path, dtype='float32', mode='w+', shape=final_shape)

    final_matrix = np.full((num_patients, max_len, 1), -1.0, dtype=np.float32)
    grouped = res_df.groupby('subject_id')

    for i, pid in enumerate(tqdm.tqdm(subject_list)):
        try:
            p_feat = grouped.get_group(pid)['time_feat'].values
            seq_len = min(len(p_feat), max_len)
            final_matrix[i, :seq_len, 0] = p_feat[:seq_len]
        except KeyError:
            continue # Already initialized with -1.0 padding

    np.save(output_path, final_matrix)
    print(f"Process finished. Standard NPY saved to: {output_path}")
    print(f"Normalization Stats: min={t_min:.4f}, max={t_max:.4f}")
    return output_path
    
# Configuration
RAW_DATA = "/home/users/nus/e1582377/MedFlow/data/raw/MIMIC-IV"
COHORT_PATH = "/home/users/nus/e1582377/MedFlow/data/processed_6/mimiciv_cohort.csv"
OUTPUT_NPY = "/home/users/nus/e1582377/MedFlow/data/processed_6/mimiciv_con_time_6.npy"

extract_continuous_time_industrial(RAW_DATA, COHORT_PATH, OUTPUT_NPY)
from typing import List, Dict, Any
from pathlib import Path
import logging

import pandas as pd 
from omegaconf import DictConfig
from hydra.utils import call, instantiate

logger = logging.getLogger("emtbench")

from eflbench.jobs import run_job
from eflbench.dataset import erase_dataset_fs_dir

def bench_one_model(model_name: str, model_cfg: DictConfig, data_cfg: DictConfig, job_cfg: DictConfig):
    """Benchmark one model."""

    logger.info(f"--------------------------------------------:)-----------------------------------------------")
    logger.info(f"Benchmarking {model_name} ...")
    logger.info(f"{model_cfg = }")
    
    # Get dataloader
    dataloader = call(data_cfg.prepare)

    # run the job
    job_metrics = {}
    try:
        job_metrics = run_job(model_cfg, dataloader, job_cfg=job_cfg)
    except Exception as ex:
        logger.warning(ex)

    # super summary of what was run
    conf = {**{k:v for k, v in data_cfg.items() if k != 'prepare'}, **model_cfg.stats}

    return job_metrics, conf


def save_entry(results: Dict, filename: Path):
    """Add result to pickled dataframe if exist, else create for the first time."""

    df = pd.DataFrame.from_dict([results])
    if filename.exists():
        df_prev = pd.read_pickle(filename)
        df = pd.concat([df_prev, df], ignore_index=True)
    
    # write
    df.to_pickle(filename)
    logger.info(df)


def run(bench_cfg: DictConfig, save_path: str):

    
    results_pkl_filename = Path(save_path)/"results_df.pkl"

    if 'path_to_data' in bench_cfg.dataset_fn:
        erase_dataset_fs_dir(bench_cfg.dataset_fn.path_to_data)


    for model_name, model_cfg in bench_cfg.model.items():
        bench_results, basic_bench_conf = bench_one_model(model_name, model_cfg, data_cfg=bench_cfg.dataset, job_cfg=bench_cfg.job)

        # embedd essential settings
        results_entry = {**bench_results, 'model': model_name, **basic_bench_conf}
        # save results of model
        save_entry(results_entry, results_pkl_filename)

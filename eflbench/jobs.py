import logging
from time import time
from omegaconf import DictConfig
from hydra.utils import call, instantiate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("emtbench")

def train(net:nn.Module, dataloader, criterion: DictConfig, device: str, optim: DictConfig, max_steps: int):
    """Train the network."""
    logger.info(f"Started train()")

    # instantiate optimizer
    optim = instantiate(optim, params=filter(lambda p: p.requires_grad, net.parameters()))
    criterion = instantiate(criterion)
    
    net.to(device)
    net.train()
    t_start = time()
    batch_count = 0
    with tqdm(total=len(dataloader.dataset)) as t:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optim.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optim.step()
            t.update(images.shape[0])
            batch_count += 1
            if batch_count == max_steps:
                break
    
    logger.info(f"Ended train()")
    return {'t_train': time() - t_start, 'train_batches': batch_count}

def finetune(net, dataloader, criterion, device: str, optim: DictConfig, max_steps: int):
    """Freeze most of the model, then call standard train."""

    net.set_for_finetuning()

    metrics = train(net, dataloader, criterion, device, optim, max_steps)
    net.disable_finetune()

    tt = metrics.pop('t_train')
    t_batches = metrics.pop('train_batches')
    return {'t_finetune': tt, 'finetune_batches': t_batches, **metrics}

def evaluate(net, dataloader, criterion, device: str, max_steps: int, **kwargs):
    """Validate the network"""
    logger.info(f"Started evaluate()")
    correct, total, loss = 0, 0, 0.0
    net.to(device)
    net.eval()
    t_start = time()
    batch_count = 0
    with torch.no_grad():
        with tqdm(total=len(dataloader.dataset)) as t:
            for data in dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                batch_count += 1
                t.update(images.shape[0])
                if batch_count == max_steps:
                    break

    logger.info(f"Ended evaluate()")
    return {'t_evaluate': time() - t_start, 'evaluate_batches': batch_count}
    

def run_job(model_cfg: DictConfig, dataloader: DataLoader, job_cfg: DictConfig):
    
    logger.info(job_cfg.tasks)

    # warmup
    if job_cfg.do_warmup:
        logger.info('Warming up....')
        # Instantiate the model
        model = instantiate(model_cfg.model)
        _ = call(job_cfg.warmup, net=model, dataloader=dataloader)

    # taks
    metrics = {}
    for _, task in job_cfg.tasks.items():
        model = instantiate(model_cfg.model)
        task_metrics = call(task, net=model, dataloader=dataloader)
        metrics = {**metrics, **task_metrics}

    return metrics

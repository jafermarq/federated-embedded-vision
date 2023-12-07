from typing import Dict, Optional

import logging

import torch.nn as nn

logger = logging.getLogger("emtbench")

class BenchModel(nn.Module):
    def __init__(self, build: nn.Module,
                 finetune_keys: Optional[Dict[str, str]] = None):
        super().__init__()

        self.model = build
        self.finetune_keys = finetune_keys


    def set_for_finetuning(self):
        logger.info("Setting model for finetuning ...")
        if not self.finetune_keys:
            raise ValueError("To finetune a model you must specify the set of keys" \
                             " in the model's state_dict to exclude from being frozen.")

        else:
            logger.info(f"Finetuning layers: {self.finetune_keys}")
            for name, m in self.named_parameters(): # ! Note we all names are assumed to start with 'model.'
                if name not in self.finetune_keys:
                    m.requires_grad = False
                else:
                    m.requires_grad = True

    def disable_finetune(self):
        for m in self.model.parameters():
            m.requires_grad = True

    def forward(self, x):
        return self.model(x)

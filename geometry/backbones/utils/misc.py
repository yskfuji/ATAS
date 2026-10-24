import gc
import torch


def cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

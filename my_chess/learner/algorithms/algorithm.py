from ray.rllib.algorithms import Algorithm as Algorithmtemp, AlgorithmConfig as AlgorithmConfigtemp
import torch

class Algorithm(Algorithmtemp):
    def getName(self):
        return self.__class__.__name__

class AlgorithmConfig(AlgorithmConfigtemp):
    def getName(self):
        return self.__class__.__name__

def determine_sum_dimensions(num_dims:int, batch_idx:int=-1):
    sum_dims = torch.arange(num_dims)
    if batch_idx != -1:
        sum_dims = sum_dims[sum_dims != batch_idx]
    return sum_dims

def average(metric:torch.Tensor, num_dims:int, batch_idx:int=-1, mask:torch.Tensor=None):
    sum_dims = determine_sum_dimensions(num_dims, batch_idx)
    org_shape = metric.shape
    if mask is None:
        mask = torch.tensor(org_shape)[sum_dims].prod()
    else:
        mask = mask.sum(sum_dims.tolist())
    metric = metric.sum(sum_dims.tolist()) / mask
    metric = torch.nan_to_num(metric)
    if batch_idx != -1:
        metric = metric.sum() / org_shape[batch_idx]
    return metric

def measure_accuracy(input:torch.Tensor, target:torch.Tensor, batch_idx:int=-1) -> torch.Tensor:
    acc = input == target
    acc = average(acc, input.ndim, batch_idx)
    return acc

def measure_precision(input:torch.Tensor, target:torch.Tensor, batch_idx:int=-1):
    prec_mask = input.int() == 1
    prec = torch.zeros_like(input, dtype=bool)
    prec[prec_mask] = input[prec_mask] == target[prec_mask]
    prec = average(prec, input.ndim, batch_idx, prec_mask)
    return prec

def measure_recall(input:torch.Tensor, target:torch.Tensor, batch_idx:int=-1):
    
    rec_mask = target.int() == 1
    rec = torch.zeros_like(input, dtype=bool)
    rec[rec_mask] = input[rec_mask] == target[rec_mask]
    rec = average(rec, input.ndim, batch_idx, rec_mask)
    return rec
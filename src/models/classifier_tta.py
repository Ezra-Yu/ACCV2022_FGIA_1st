from typing import List

import torch
import torch.nn as nn

from mmcls.structures import ClsDataSample

from mmengine.model import BaseTTAModel
from mmcls.registry import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class ClassifierTTA(BaseTTAModel):
    def merge_preds(self, data_samples_list: List[List[ClsDataSample]]):
        num_tta_permutations = len(data_samples_list)
        num_samplers = len(data_samples_list[0])
        results = [data_samples_list[0][j] for j in range(num_samplers)]
        
        for j in range(num_samplers):
            results[j].scores = torch.zeros_like(results.scores)
 
        for i in range(num_tta_permutations):
            for j in range(num_samplers):
                results[j].scores += data_samples_list[i][j].score / num_tta_permutations
        
        return results  

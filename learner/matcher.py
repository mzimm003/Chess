import torch, os
from torch import nn
from scipy.optimize import linear_sum_assignment

import numpy as np

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_legal_move: float = 1, focal_alpha = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
        """
        super().__init__()
        self.cost_legal_move = cost_legal_move
        assert cost_legal_move != 0 , "all costs cant be 0"

        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_class": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_class"].shape[:2]

        out_prob = outputs["pred_class"].sigmoid()  # [batch_size * num_queries, num_classes]
        tgt_ids = targets["labels"].int()

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_legal_move = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_legal_move = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        pos_cost_legal_move_all_combos = pos_cost_legal_move[[*np.indices([*pos_cost_legal_move.shape[:-1], tgt_ids.shape[-1]])[:-1], tgt_ids[:,None,:]]] 
        neg_cost_legal_move_all_combos = neg_cost_legal_move[[*np.indices([*neg_cost_legal_move.shape[:-1], tgt_ids.shape[-1]])[:-1], tgt_ids[:,None,:]]]
        cost_class = pos_cost_legal_move_all_combos - neg_cost_legal_move_all_combos


        # Final cost matrix
        C = self.cost_legal_move * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        indices = [[torch.tensor(x) for x in linear_sum_assignment(c)] for c in C]
        return torch.cat([torch.cat([i.unsqueeze(1),j.unsqueeze(1)],1).unsqueeze(0) for i,j in indices],0)
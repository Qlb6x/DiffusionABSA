import torch
from scipy.optimize import linear_sum_assignment
from .lap import auction_lap
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_span: float = 1, match_boundary_type='f1', solver="hungarian"):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_span = cost_span
        self.match_boundary_type = match_boundary_type
        self.solver = solver

    @torch.no_grad()
    def forward(self, outputs, targets):
        # print(outputs, targets)
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
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

        if self.solver == "order":
            sizes = targets["sizes"]
            indices = [(list(range(size)), list(range(size))) for size in sizes]
        else:
            # bs= 8 , num_queries conf里面的num_proposals
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            # flatten(0, 1) ->
            # softmax(dim=-1) 和 2 的效果一样，就是最后一层做softmax，求和为1
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(dim=-1)  # [batch_size * num_queries, 8]
            aspect_left = outputs["pred_left"].flatten(0, 1)  # [batch_size * num_queries, ? 57 39 57 61 56 不确定]
            aspect_right = outputs["pred_right"].flatten(0, 1)  # [batch_size * num_queries, ? 57 39 57 61 56 不确定]

            gt_ids = targets["labels"]  # [batch_size * num_queries]
            gt_left = targets["gt_left"]  # [batch_size * num_queries]
            gt_right = targets["gt_right"]  # [batch_size * num_queries]

            # import pdb;pdb.set_trace()
            cost_class = -out_prob[:, gt_ids]

            C = None

            # Final cost matrix
            # cost_span boundary的系数 ，cost_class type的系数， argmax 返回最大值索引
            if self.match_boundary_type == "f1":
                aspect_left_idx = aspect_left.argmax(dim=-1)  # [batch_size * num_queries]
                aspect_right_idx = aspect_right.argmax(dim=-1)  # [batch_size * num_queries]
                cost_dis = torch.abs(aspect_left_idx.unsqueeze(-1) - gt_left.unsqueeze(0)) + torch.abs(aspect_right_idx.unsqueeze(-1) - gt_right.unsqueeze(0))

                C = self.cost_span * cost_dis + self.cost_class * cost_class

            if self.match_boundary_type == "logp":
                # cost_span  [batch_size * num_queries]
                # cost_class [batch_size * num_queries] 不足num_queries
                cost_span = -(aspect_left[:, gt_left] + aspect_right[:, gt_right])
                C = self.cost_span * cost_span + self.cost_class * cost_class

            C = C.view(bs, num_queries, -1)

            # num_queries * 8
            sizes = targets["sizes"]
            indices = None

            # 2 * num_queries
            # [(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
            #        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            #        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            #        51, 52, 53, 54, 55, 56, 57, 58, 59], dtype=int64)
            # , array([ 2, 31, 24, 10, 14, 17, 19, 23, 55, 29, 21, 40, 18, 36, 44, 33, 50,
            #        53, 30, 12, 22, 16,  8, 37, 11,  0,  3, 56, 34,  6,  9, 46, 41, 39,
            #        35, 25, 32, 51, 13, 57,  5,  7, 20, 58, 43, 26, 42, 47, 15,  1,  4,
            #        45, 59, 52, 38, 28, 49, 27, 54, 48],

            if self.solver == "hungarian":
                # C: torch.Size([8, 60, 480])
                C = C.cpu()
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

            if self.solver == "auction":
                indices = [auction_lap(c[i])[:2] for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

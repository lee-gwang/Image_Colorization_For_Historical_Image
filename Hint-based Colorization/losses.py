"""
Implements the knowledge distillation loss
"""
import torch.nn as nn
import torch
from torch.nn import functional as F

class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, input, target):
        mask = torch.zeros_like(input)
        mann = torch.abs(input - target)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl * mask + self.delta * (mann - .5 * self.delta) * (1 - mask)
        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=-1, keepdim=False).mean()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, input, target):
        return torch.sum(torch.abs(input - target), dim=-1, keepdim=False).mean()


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, input, target):
        return torch.sum((input - target)**2, dim=-1, keepdim=False).mean()

class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, reconstruct_weight=1.0, sparsity_weight=1.0, distill_weight=1.0, mse_token=False, flops_ratio=1, flops_ratio2=1):
        super().__init__()
        # teacher
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.flops_ratio = flops_ratio
        self.flops_ratio2 = flops_ratio2
        self.mse_token = mse_token

        # weight loss
        self.reconstruct_weight = reconstruct_weight
        self.sparsity_weight = sparsity_weight
        self.distill_weight = distill_weight

    def forward(self, inputs, outputs, mask, labels):
        """
        Args:
            # inputs: The original inputs that are feed to the teacher model
            # outputs: the outputs of the model to be trained. It is expected to be
            #     either a Tensor, or a Tuple[Tensor, Tensor], with the original output
            #     in the first position and the distillation predictions as the second output
            # labels: the labels for the base criterion

            inputs : input images
            outputs : (features, outputs)
            labels 
        """

        # pred, token_pred, mask, out_pred_score = outputs
        features, output = outputs
        layer_mask , token_mask, bool_hinted_pos = mask

        # sparsity loss
        sparsity_loss = 0.0
        sparsity_loss += (layer_mask - self.flops_ratio)**2
        sparsity_loss += (token_mask - self.flops_ratio2)**2

        reconstruct_loss = self.base_criterion(output, labels)

        with torch.no_grad():
            token_t = self.teacher_model(inputs, bool_hinted_pos) # token_t --> features


        # B, N, C = token_pred.size()
        # assert mask.numel() == B * N

        # bool_mask = mask.reshape(B*N) > 0.5

        # loss_part = []

        # token_pred = token_pred.reshape(B*N, C)
        # token_t = token_t.reshape(B*N, C)

        # if mask.sum() < 0.1:
        #     token_kl_loss = token_pred.new(1,).fill_(0.0)
        # else:
        #     token_t = token_t[bool_mask]
        #     token_pred = token_pred[bool_mask]
            # if self.mse_token:
            #     token_kl_loss = torch.pow(token_pred - token_t, 2).mean()
            # else:
            #     token_kl_loss = F.kl_div(
            #             F.log_softmax(token_pred, dim=-1),
            #             F.log_softmax(token_t, dim=-1),
            #             reduction='batchmean',
            #             log_target=True
            #         )

        if self.mse_token:
            token_kl_loss = torch.pow(features - token_t, 2).mean()
            # token_kl_loss = torch.pow(F.tanh(features) - token_t, 2).mean()

        else:
            token_kl_loss = F.kl_div(
                    F.log_softmax(features, dim=-1),
                    F.log_softmax(token_t, dim=-1),
                    reduction='batchmean',
                    log_target=True
                )
        # huber loss + sparsity loss + distill reconstruct loss
        loss = self.reconstruct_weight * reconstruct_loss + self.sparsity_weight * sparsity_loss  + self.distill_weight * token_kl_loss

        return loss


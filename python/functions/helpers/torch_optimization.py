from torch import sigmoid
import numpy as np
import torch
from torch import nn


def coverage_loss(w, d):
    w_torch = sigmoid(w)

    v = w_torch.view(-1, 1, 1, 1)  # same number of dimension as *t*
    voxels_torch = v * d

    sum_all_voxels_torch = torch.sum(d, dim=(1, 2, 3))
    cov_term_torch = torch.sum(sum_all_voxels_torch * w_torch)  # cov_term

    cov_term_torch = torch.exp(-cov_term_torch)
    cov_term_torch = cov_term_torch + torch.sum(w_torch)

    weighted_torch = torch.sum(voxels_torch, dim=0)  # weighted for each voxel
    avg_cov_torch = torch.mean(weighted_torch)

    weighted_torch = weighted_torch - avg_cov_torch
    avg_term_torch = torch.sum(torch.abs(weighted_torch))

    loss = cov_term_torch * 1.0 + avg_term_torch * 1.0
    return loss


def occlusion_loss(w, gauss, rescanning_weights):
    sig_w = sigmoid(w)
    seen_penalty = sig_w * rescanning_weights
    seen_penalty_term = torch.sum(seen_penalty)
    sim_penalty_term = torch.sum(sig_w + gauss)
    return 5.0 * sim_penalty_term + 5.0 * seen_penalty_term


class OcclusionPrevention(nn.Module):
    def __init__(self, d, gauss_rep, rescanning_weights):
        super().__init__()

        self.d = torch.nn.Parameter(data=torch.Tensor(d).data, requires_grad=False)
        self.gauss = torch.nn.Parameter(data=torch.Tensor(gauss_rep).data, requires_grad=False)
        self.rescanning_weights = torch.nn.Parameter(data=torch.Tensor(rescanning_weights).data, requires_grad=False)

        self.weights = torch.nn.Parameter(data=torch.Tensor((d.shape[0])), requires_grad=True)
        self.weights.data.uniform_(-1, 1)

    def forward(self):
        """ The cost function for occlusion prevention."""

        cost_occlusion = occlusion_loss(self.weights, self.gauss, self.rescanning_weights)
        cost_coverage = coverage_loss(self.weights, self.d)

        return cost_occlusion + cost_coverage


class CoverageModel(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, d):
        super().__init__()
        # initialize weights with random numbers

        self.d = torch.nn.Parameter(data=torch.Tensor(d).data, requires_grad=False)

        self.weights = torch.nn.Parameter(data=torch.Tensor((d.shape[0])), requires_grad=True)
        self.weights.data.uniform_(-1, 1)

        # make weights torch parameters

    def forward(self):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        return coverage_loss(self.weights, self.d)


def optimization_loop(model, optimizer, args, n=1000):
    " Training loop for torch model. "
    losses = []
    for i in range(n):
        loss = model(args)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
    return losses, model.weights


def pytorch_minimize(model, w, d, nr_angles, nr_rows, nr_cols):
    """
    The function assumes that d is a dictionary where poses are saved as:
    d = {(angle, row, col) : array()}.

    :return:
    """

    # Bringing w and d expressed in consistent ways.
    d_array = np.array([d[item] for item in d.keys()])

    optimizer = torch.optim.Adam([w], lr=1)
    model = CoverageModel(d_array)
    losses, weights = optimization_loop(model, optimizer)





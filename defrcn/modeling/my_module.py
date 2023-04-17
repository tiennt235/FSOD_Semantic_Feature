#!/usr/bin/env python
#
# OPTIMAL TRANSPORT NODE
# Implementation of differentiable optimal transport using implicit differentiation. Makes use of Sinkhorn normalization
# to solve the entropy regularized problem (Cuturi, NeurIPS 2013) in the forward pass. The problem can be written as
# Let us write the entropy regularized optimal transport problem in the following form,
#
#    minimize (over P) <P, M> + 1/\gamma KL(P || rc^T)
#    subject to        P1 = r and P^T1 = c
#
# where r and c are m- and n-dimensional positive vectors, respectively, each summing to one. Here m-by-n matrix M is
# the input and m-by-n dimensional positive matrix P is the output. The above problem leads to a solution of the form
#
#   P_{ij} = \alpha_i \beta_j e^{-\gamma M_{ij}}
#
# where \alpha and \beta are found by iteratively applying row and column normalizations.
#
# We also provide an option to parametrize the input in log-space as M_{ij} = -\log Q_{ij} where Q is a positive matrix.
# The matrix Q becomes the input. This is more numerically stable for inputs M with large positive or negative values.
#
# See accompanying Jupyter Notebook at https://deepdeclarativenetworks.com.
#
# Stephen Gould <stephen.gould@anu.edu.au>
# Dylan Campbell <dylan.campbell@anu.edu.au>
# Fred Zhang <frederic.zhang@anu.edu.au>
#

# https://github.com/anucvml/ddn/blob/master/ddn/pytorch/optimal_transport.py

import torch.nn.functional as F
import torch
import torch.nn as nn
import warnings
from detectron2.data import MetadataCatalog, DatasetCatalog


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    """
    PyTorch function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector 1/H
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector 1/W

    You can back propagate through this function in O(TBWH) time where T is the number of iterations taken to converge.
    """

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.min(M, 2, keepdim=True)[0]))

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        # if torch.max(torch.abs(beta - c)) <= eps:
        #     break
        P = P / beta.view(B, 1, W) * c

    return P


def _sinkhorn_inline(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    """As above but with inline calculations for when autograd is not needed."""
    # eps = 1e-2
    eps2 = 1e4
    eps3 = 0  # 1e-6
    B, H, W = M.shape
    assert r is None or r.shape == (
        B, H) or r.shape == (1, H), f'{r.shape}, {B, H}'
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0), f'{torch.all(M > 0.0)}'
    # print(torch.all(M > 0.0))
    r = r.unsqueeze(dim=2)
    c = c.unsqueeze(dim=1)

    debug = False
    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.min(M, 2, keepdim=True)[0]))
    # P = torch.clamp(P, min=1e-3, max=eps2)
    if debug:
        print('first P:', P.max(), P.min())
        if torch.any(torch.isnan(P)):
            print('first P isnan :', torch.where(torch.isnan(P)))
            assert 2 == 4

    for i in range(maxiters):
        # print('first: ',P)
        alpha = torch.sum(P, 2) + eps3
        alpha[torch.where(alpha == 0)] = eps

        # Perform division first for numerical stability
        if debug:
            print('computed alpha:', alpha.max(), alpha.min())
        if debug and torch.any(alpha == 0):
            ind = torch.where(alpha == 0)
            print(f'iszeros alpha at {i}:', ind)
            print('value of P: ', P[ind[0], ind[1]])
            assert 2 == 1
            break

        if debug and torch.any(torch.isnan(alpha)):
            ind = torch.where(torch.isnan(alpha))
            print(P[ind[0], ind[1]])
            print('isnan alpha:', ind)
            assert 2 == 1
            break

        P /= alpha.view(B, H, 1)
        P *= r

        if debug:
            print('after cal P follow r:', P.max(), P.min())
            if torch.any(torch.isnan(P)):
                ind = torch.where(torch.isnan(P))
                print('isnan P second:', ind)
                print('alpha:', alpha[ind[0], ind[1]])
                print('P:', P[ind[0], ind[1]])
                assert 2 == 1

        beta = torch.sum(P, 1) + eps3
        beta[torch.where(beta == 0)] = eps
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P /= beta.view(B, 1, W)
        P *= c

        if debug:
            print('computed beta:', beta.max(), beta.min())
            print('after cal P follow c:', P.max(), P.min())
            if torch.any(torch.isnan(beta)):
                ind = torch.where(torch.isnan(beta))
                print(P[ind[0], ind[1]])
                print('isnan beta:', ind)
                assert 2 == 1
                break

        # print('Third: ',P)
        # print('-'*69)
        # if torch.any(torch.isnan(Pre)):
        #     break

        # P=Pre
        # P = torch.clamp(P, min=-eps2, max=eps2)
        # del Pre

        if debug and torch.any(torch.isnan(P)):
            ind = torch.where(torch.isnan(P))
            print('isnan P:', torch.where(torch.isnan(P)))
            print('isnan alpha:', torch.where(torch.isnan(alpha)))
            print('isnan alpha:', alpha[ind[0], ind[1]])
            print('isnan beta:', torch.where(torch.isnan(beta)))
            print('isnan beta:', beta[ind[0], ind[1]])
            print(P)
            print(alpha)
            print(beta)
            print('-'*24)

            print('zeros P:', torch.where(P == 0))
            print('isnan P:', torch.where(torch.isnan(P)))
            print('isinf P:', torch.where(torch.isinf(P)))
            assert 2 == 1
            break
        if debug:
            print('------', end='\n'*2)

    if debug and torch.any(torch.isnan(P)):
        print('isnan P:', torch.where(torch.isnan(P)))
    # print(P)
    return P


class OptimalTransportFcn(torch.autograd.Function):
    """
    PyTorch autograd function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector

    Allows for approximate gradient calculations, which is faster and may be useful during early stages of learning,
    when exp(-\gamma M) is already nearly doubly stochastic, or when gradients are otherwise noisy.

    Both r and c must be positive, if provided. They will be normalized to sum to one.
    """

    @staticmethod
    def forward(ctx, M, r=None, c=None, gamma=0.5, eps=1.0e-6, maxiters=1000, logspace=False, method='block'):
        """Solve optimal transport using skinhorn. Method can be 'block', 'full' or 'approx'."""
        assert method in ('block', 'full', 'approx')

        with torch.no_grad():
            # normalize r and c to ensure that they sum to one (and save normalization factor for backward pass)
            if r is not None:
                ctx.inv_r_sum = 1.0 / torch.sum(r, dim=1, keepdim=True)
                r = ctx.inv_r_sum * r
            if c is not None:
                ctx.inv_c_sum = 1.0 / torch.sum(c, dim=1, keepdim=True)
                c = ctx.inv_c_sum * c
            # print('r: ', r)
            # print('c: ', c)
            # run sinkhorn
            P = _sinkhorn_inline(M, r, c, gamma, eps, maxiters, logspace)

        ctx.save_for_backward(M, r, c, P)
        ctx.gamma = gamma
        ctx.logspace = logspace
        ctx.method = method
        return P

    @staticmethod
    def backward(ctx, dJdP):
        """Implement backward pass using implicit differentiation."""
        debug = False
        M, r, c, P = ctx.saved_tensors
        B, H, W = M.shape

        # initialize backward gradients (-v^T H^{-1} B with v = dJdP and B = I or B = -1/r or B = -1/c)
        dJdM = -1.0 * ctx.gamma * P * dJdP
        dJdr = None if not ctx.needs_input_grad[1] else torch.zeros_like(r)
        dJdc = None if not ctx.needs_input_grad[2] else torch.zeros_like(c)
        times = 1
        # return approximate gradients
        if ctx.method == 'approx':
            # print(times)
            if debug:
                print('dJdM: ', dJdM.min(), dJdM.max())
                print('dJdr: ', dJdr)
                print('dJdc: ', dJdc)
                assert 2 == 1
            return dJdM, dJdr, dJdc, None, None, None, None, None, None

        # compute exact row and column sums (in case of small numerical errors or forward pass not converging)
        alpha = torch.sum(P, dim=2)
        beta = torch.sum(P, dim=1)

        # compute [vHAt1, vHAt2] = v^T H^{-1} A^T as two blocks
        vHAt1 = torch.sum(dJdM[:, 1:H, 0:W], dim=2)
        vHAt2 = torch.sum(dJdM, dim=1)

        # compute [v1, v2] = -v^T H^{-1} A^T (A H^{-1] A^T)^{-1}
        if ctx.method == 'block':
            # by block inverse of (A H^{-1] A^T)
            PdivC = P[:, 1:H, 0:W] / beta.view(B, 1, W)
            RminusPPdivC = torch.diag_embed(
                alpha[:, 1:H]) - torch.einsum("bij,bkj->bik", P[:, 1:H, 0:W], PdivC)
            try:
                block_11 = torch.cholesky(RminusPPdivC)
            except:
                # block_11 = torch.ones((B, H-1, H-1), device=M.device, dtype=M.dtype)
                block_11 = torch.eye(
                    H - 1, device=M.device, dtype=M.dtype).view(1, H - 1, H - 1).repeat(B, 1, 1)
                for b in range(B):
                    try:
                        block_11[b, :, :] = torch.cholesky(
                            RminusPPdivC[b, :, :])
                    except:
                        # keep initialized values (gradient will be close to zero)
                        warnings.warn(
                            "backward pass encountered a singular matrix")
                        pass

            block_12 = torch.cholesky_solve(PdivC, block_11)
            block_22 = torch.diag_embed(
                1.0 / beta) + torch.einsum("bji,bjk->bik", block_12, PdivC)

            v1 = torch.cholesky_solve(vHAt1.view(
                B, H - 1, 1), block_11).view(B, H - 1) - torch.einsum("bi,bji->bj", vHAt2, block_12)
            v2 = torch.einsum("bi,bij->bj", vHAt2, block_22) - \
                torch.einsum("bi,bij->bj", vHAt1, block_12)

        else:
            # by full inverse of (A H^{-1] A^T)
            AinvHAt = torch.empty((B, H + W - 1, H + W - 1),
                                  device=M.device, dtype=M.dtype)
            AinvHAt[:, 0:H - 1, 0:H - 1] = torch.diag_embed(alpha[:, 1:H])
            AinvHAt[:, H - 1:H + W - 1, H - 1:H +
                    W - 1] = torch.diag_embed(beta)
            AinvHAt[:, 0:H - 1, H - 1:H + W - 1] = P[:, 1:H, 0:W]
            AinvHAt[:, H - 1:H + W - 1, 0:H -
                    1] = P[:, 1:H, 0:W].transpose(1, 2)
            # print(torch.inverse(AinvHAt))
            v = torch.einsum(
                "bi,bij->bj", torch.cat((vHAt1, vHAt2), dim=1), torch.inverse(AinvHAt))
            # v = torch.einsum("bi,bij->bj", torch.cat((vHAt1, vHAt2), dim=1), torch.cholesky_inverse(AinvHAt))
            v1 = v[:, 0:H - 1]
            v2 = v[:, H - 1:H + W - 1]

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} A H^{-1} B - v^T H^{-1} B
        dJdM[:, 1:H, 0:W] -= v1.view(B, H - 1, 1) * P[:, 1:H, 0:W]
        dJdM -= v2.view(B, 1, W) * P

        # multiply by derivative of log(M) if in log-space
        if ctx.logspace:
            dJdM /= -1.0 * M

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} (A H^{-1} B - C) - v^T H^{-1} B
        if dJdr is not None:
            dJdr = ctx.inv_r_sum.view(r.shape[0], 1) / ctx.gamma * \
                (torch.sum(r[:, 1:H] * v1, dim=1, keepdim=True) -
                 torch.cat((torch.zeros(B, 1, device=r.device), v1), dim=1))

        # compute v^T H^{-1} A^T (A H^{-1] A^T)^{-1} (A H^{-1} B - C) - v^T H^{-1} B
        if dJdc is not None:
            dJdc = ctx.inv_c_sum.view(
                c.shape[0], 1) / ctx.gamma * (torch.sum(c * v2, dim=1, keepdim=True) - v2)
        # print(dJdM, dJdr, dJdc)
        # return gradients (None for gamma, eps, maxiters and logspace)
        if debug:
            print('dJdM: ', dJdM)
            print('dJdr: ', dJdr)
            print('dJdc: ', dJdc)
            assert 2 == 1
        return dJdM, dJdr, dJdc, None, None, None, None, None, None


class OptimalTransportLayer(nn.Module):
    """
    Neural network layer to implement optimal transport.

    Parameters:
    -----------
    gamma: float, default: 1.0
        Inverse of the coeffient on the entropy regularisation term.
    eps: float, default: 1.0e-6
        Tolerance used to determine the stop condition.
    maxiters: int, default: 1000
        The maximum number of iterations.
    logspace: bool, default: False
        If `True`, assumes that the input is provided as \log M
        If `False`, assumes that the input is provided as M (standard optimal transport)
    method: str, default: 'block'
        If `approx`, approximate the gradient by assuming exp(-\gamma M) is already nearly doubly stochastic.
        It is faster and could potentially be useful during early stages of training.
        If `block`, exploit the block structure of matrix A H^{-1] A^T.
        If `full`, invert the full A H^{-1} A^T matrix without exploiting the block structure
    """

    def __init__(self, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False, method='block'):
        super(OptimalTransportLayer, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.maxiters = maxiters
        self.logspace = logspace
        self.method = method

    def forward(self, M, r=None, c=None):
        """
        Parameters:
        -----------
        M: torch.Tensor
            Input matrix/matrices of size (H, W) or (B, H, W)
        r: torch.Tensor, optional
            Row sum constraint in the form of a 1xH or BxH matrix. Are assigned uniformly as 1/H by default.
        c: torch.Tensor, optional
            Column sum constraint in the form of a 1xW or BxW matrix. Are assigned uniformly as 1/W by default.

        Returns:
        --------
        torch.Tensor
            Normalised matrix/matrices, with the same shape as the inputs
        """
        M_shape = M.shape
        # Check the number of dimensions
        ndim = len(M_shape)
        if ndim == 2:
            M = M.unsqueeze(dim=0)
        elif ndim != 3:
            raise ValueError(
                f"The shape of the input tensor {M_shape} does not match that of an matrix")

        # Handle special case of 1x1 matrices
        nr, nc = M_shape[-2:]
        if nr == 1 and nc == 1:
            P = torch.ones_like(M)
        else:
            P = OptimalTransportFcn.apply(
                M, r, c, self.gamma, self.eps, self.maxiters, self.logspace, self.method)
        return P.view(*M_shape)


def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params["alpha"]
    T = params["temperature"]
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def loss_fn_kd_only(outputs, labels, bg_label, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params["alpha"]
    T = params["temperature"]
    reduction = 'none'
    # reduction = 'batchmean'
    KD_loss = nn.KLDivLoss(reduction=reduction)(F.log_softmax(outputs/T, dim=1),
                                                F.softmax(teacher_outputs/T, dim=1),)
    if reduction == 'none':
        KD_loss = KD_loss.sum(1)
        # print('KD_loss', KD_loss.shape)
        # print('labels', labels.shape)

        bg_selection_mask = labels == bg_label
        bg_idxs = bg_selection_mask.nonzero().squeeze(1)

        KD_loss[bg_idxs] = KD_loss[bg_idxs]*1.5
        KD_loss = KD_loss.sum()/(labels.shape[0])

    # print('bg_idxs', bg_idxs.shape)
    # print(bg_idxs)
    # print(torch.where(labels=bg_label))

    return KD_loss*T*T * alpha


def blogits_matrix(a, b):
    """
    added eps for numerical stability
    """
    logits_mt = torch.einsum('bik,bjk->bij', a, b)
    logits_mt = torch.max(logits_mt) - logits_mt
    return logits_mt


def bsim_matrix(a, b, tau=1):
    """
    added eps for numerical stability
    """
    norm = torch.nn.functional.normalize
    a_norm, b_norm = norm(a, dim=-1), norm(b, dim=-1)
    sim_mt = torch.einsum('bik,bjk->bij', a_norm, b_norm)*tau
    # sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt


def sim_matrix(a, b, eps=1e-12):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))

    return sim_mt


@torch.no_grad()
def Sinkhorn(K, u, v, max_iter=10, **arg):
    K = torch.exp(-K / 0.05)
    # print('K:', K)
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-1
    for _ in range(max_iter):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(),
                             r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T


def _has_nan_or_inf(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


@torch.no_grad()
def sinkhorn_fb(cost_mat, r_prob=None, c_prob=None, eps=0.05, niter=10):
    """
    cost_mat: s1, s2, ..., sn, M, N
    r_prob: s1, s2, ..., sn, M
    c_prob: s1, s2, ..., sn, N
    """
    Q = torch.exp(-cost_mat / eps)
    Q = Q / Q.sum(dim=[-2, -1], keepdim=True)
    M, N = Q.shape[-2], Q.shape[-1]

    if r_prob is not None:
        # s1, ..., sn, M -> s1, ..., sn, M, 1
        r_prob = (r_prob / r_prob.sum(dim=-1, keepdim=True)).unsqueeze(-1)
        assert not _has_nan_or_inf(r_prob)
    else:
        r_prob = 1 / M

    if c_prob is not None:
        # s1, ..., sn, N -> s1, ..., sn, 1, N
        c_prob = (c_prob / c_prob.sum(dim=-1, keepdim=True)).unsqueeze(-2)
        assert not _has_nan_or_inf(c_prob)
    else:
        c_prob = 1 / N

    for _ in range(niter):
        # normalize each row: total weight per row must be r_prob
        Q /= Q.sum(dim=-1, keepdim=True)
        Q *= r_prob
        # normalize each column: total weight per column must be c_prob
        Q /= Q.sum(dim=-2, keepdim=True)
        Q *= c_prob
    return Q


@torch.no_grad()
def ipot_WD(C, a1, a2, beta=0.05, max_iter=10, L=1, use_path=True, return_map=True, return_loss=True):
    u"""
    Solve the optimal transport problem and return the OT matrix
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F 
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - a and b are source and target weights (sum to 1)
    The algorithm used priximal point method
    Parameters
    ----------
    a1 : torch.ndarray (ns,)
        samples weights in the source domain
    a2 : torch.ndarray (nt,) or torch.ndarray (nt,nbb)
        samples in the target domain, compute sinkhorn with multiple targets
        and fixed M if b is a matrix (return OT loss + dual variables in log)
    C : torch.ndarray (ns,nt)
        loss matrix
    beta : float, optional
        Step size of poximal point iteration
    max_iter : int, optional
        Max number of iterations
    L : int, optional
        Number of iterations for inner optimization
    use_path : bool, optional
        Whether warm start method is used
    return_map : bool, optional
        Whether the optimal transportation map is returned
    return_loss : bool, optional
        Whether the list of calculated WD is returned
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    loss : list
        log of loss (Wasserstein distance)
    Examples
    --------
    >>> import ipot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ipot.ipot_WD(a,b,M,beta=1)
    array([[ 1.,  0.],
           [ 0.,  1.]])
    References
    ----------
    [1] Xie Y, Wang X, Wang R, et al. A Fast Proximal Point Method for 
    Wasserstein Distance[J]. arXiv preprint arXiv:1802.04307, 2018.


    """

    n1 = len(a1)
    n2 = len(a2)
    v = torch.ones([n2, ]).cuda()
    u = torch.ones([n1, ]).cuda()

    P = torch.ones((n1, n2)).cuda()/(n1*n2)

    K = torch.exp(-(C/beta))
    if return_loss == True:
        loss = [torch.sum(P*C)]

    for outer_i in range(max_iter):
        Q = K*P

        if use_path == False:
            v = torch.ones([n1, ]).cuda()
            u = torch.ones([n2, ]).cuda()

        for i in range(L):
            u = a1/torch.matmul(Q, v)
            v = a2/torch.matmul(Q.T, u)
        # P = np.expand_dims(u,axis=1)*Q*np.expand_dims(v,axis=0)
        P = torch.unsqueeze(u, axis=1)*Q*torch.unsqueeze(v, axis=0)

        if return_loss == True:
            W = torch.sum(P*C)
            loss.append(W)

    if return_loss == True:
        if return_map == True:
            return P, loss

        else:
            return loss

    else:
        if return_map == True:
            return P

        else:
            return None


class memory_bank(torch.nn.Module):
    def __init__(self, num_classes=60, capacity=32, input_dim=1024, device='cpu', mem=None):
        super(memory_bank, self).__init__()
        # if mem:
        self.fixed_memory = mem
        # self.memory = torch.nn.Parameter(torch.randn(
        #     num_classes, capacity, input_dim), requires_grad=False)

        self.memory = torch.randn(
            num_classes, capacity, input_dim).to(device)  # , requires_grad=False)

        self.cap = capacity
        self.num_classes = num_classes
        self.device = device

    @torch.no_grad()
    def get_mem(self, cls):
        return self.memory[cls]

    def forward(self, x, classes):
        # self.update3(x, classes)
        self.update4(x, classes)
        # self.update5(x, classes)
        return self.memory

    @torch.no_grad()
    def update4(self, instances, classes):
        # print([i for i in self.fixed_memory.items()][0])
        fixed_ins = torch.cat([i for i in self.fixed_memory.values()], dim=0)
        # print(fixed_ins.shape)
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        k = 0.99  # 0.9  # 0.5 ~ 55% for x2 schedule
        for cls in __unique:
            if cls == self.num_classes:
                continue
            # fixed_id = torch.randperm(fixed_ins.shape[0])[:self.cap]

            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            new_ins = torch.cat([instances[index[0]], candidates])[:self.cap]

            t = torch.einsum('qk,mk->qm', new_ins, fixed_ins)

            index = t.sort(dim=1, descending=False)[1][:, :1].squeeze(-1)
            # print(t.sort(dim=1, descending=False)[0])
            # print('index:', index.shape)
            # print('fixed_ins:', fixed_ins[index].shape)

            self.memory[cls] = k*new_ins + (1-k)*(fixed_ins[index])
            del new_ins, candidates

    @torch.no_grad()
    def update5(self, instances, classes):
        # print([i for i in self.fixed_memory.items()][0])
        fixed_ins = torch.cat([i for i in self.fixed_memory.values()], dim=0)
        # print(fixed_ins.shape)
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        k = 0.99  # 0.9  # 0.5 ~ 55% for x2 schedule
        for cls in __unique:
            if cls == self.num_classes:
                continue
            # fixed_id = torch.randperm(fixed_ins.shape[0])[:self.cap]

            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            new_ins = torch.cat([instances[index[0]], candidates])[:self.cap]

            t = torch.einsum('qk,mk->qm', new_ins, fixed_ins)

            index = t.sort(dim=1, descending=False)[1][:, :5]
            # print(t.sort(dim=1, descending=False)[0])
            # print(fixed_ins[index].mean(1).shape)
            # print('index:', index.shape)
            # print('fixed_ins:', fixed_ins[index].shape)

            self.memory[cls] = k*new_ins + (1-k)*(fixed_ins[index].mean(1))
            del new_ins, candidates

    @torch.no_grad()
    def update3(self, instances, classes):
        fixed_ins = []
        for id in torch.randperm(len(self.fixed_memory))[:5]:
            # torch.randperm()
            ins = self.fixed_memory[int(id)]
            fixed_ins.append(ins)

        fixed_ins = torch.cat(fixed_ins, dim=0)

        assert len(fixed_ins) >= self.cap

        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        k = 0.99  # 0.9  # 0.5 ~ 55% for x2 schedule
        for cls in __unique:
            if cls == self.num_classes:
                continue
            fixed_id = torch.randperm(fixed_ins.shape[0])[:self.cap]

            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            new_ins = torch.cat([instances[index[0]], candidates])[:self.cap]

            # t = torch.einsum('qk,mk->qm', fixed_ins[fixed_id], new_ins)
            # t = torch.nn.functional.softmax(t, dim=1)

            self.memory[cls] = k*new_ins + (1-k)*(fixed_ins[fixed_id])
            del new_ins, candidates

    @torch.no_grad()
    def update2(self, instances, classes):
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()
        k = 0.2
        for cls in __unique:
            # if cls == self.num_classes:
            #     continue
            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            ins = instances[index[0]]

            # ins_norm = torch.nn.functional.normalize(ins)
            # can_norm = torch.nn.functional.normalize(candidates)

            # ins = instances[index[0]]
            # t = torch.einsum('qk,mk->qm', ins_norm, can_norm)
            t = torch.einsum('qk,mk->qm', ins, candidates)
            t = torch.nn.functional.softmax(t, dim=1)

            _, sim_index = t.sort()
            sim_index = sim_index[:, 0]

            candidates[sim_index] = (1-k)*candidates[sim_index] + k*ins
            # candidates = (1-k)*candidates + k*(t.T@ins)
            self.memory[cls] = candidates

            del candidates

    @torch.no_grad()
    def update(self, instances, classes):
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        for cls in __unique:
            if cls == self.num_classes:
                continue
            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            new_ins = torch.cat([instances[index[0]], candidates])
            self.memory[cls] = new_ins[:self.cap]
            del new_ins, candidates


class memory_bank_ot(torch.nn.Module):
    def __init__(self, num_classes=60, capacity=32, input_dim=1024, device='cpu', mem=None, cfg=None):
        super(memory_bank_ot, self).__init__()

        self.fixed_memory = mem
        self.memory = torch.randn(
            num_classes, capacity, input_dim).to(device)

        self.cap = capacity
        self.num_classes = num_classes
        self.device = device
        self.few_shot_mode = False
        if cfg:
            few_shot_dataset = cfg.DATASETS.TRAIN[0]
            if 'shot' in few_shot_dataset:
                self.few_shot_mode = True

                # meta = MetadataCatalog.get(few_shot_dataset)
                # self.novel_pos = meta.novel_classes
                # print(meta.thing_classes)
                # print(self.novel_pos)
                # assert 0

    @torch.no_grad()
    def get_mem(self, cls):
        return self.memory[cls]

    def forward(self, x, classes):
        self.update(x, classes)
        # if self.few_shot_mode:
        #     return self.memory[-5:]

        return self.memory

    @torch.no_grad()
    def update(self, instances, classes):
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        for cls in __unique:
            if cls == self.num_classes:
                continue
            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            new_ins = torch.cat([instances[index[0]], candidates])
            self.memory[cls] = new_ins[:self.cap]
            del new_ins, candidates


class memory_bank_ot2(torch.nn.Module):
    def __init__(self, num_classes=60, capacity=32, input_dim=1024, device='cpu', mem=None, cfg=None):
        super(memory_bank_ot2, self).__init__()

        self.fixed_memory = mem
        self.memory = torch.randn(capacity, input_dim).to(device)

        self.cap = capacity
        self.num_classes = num_classes
        self.device = device
        self.few_shot_mode = False
        self.mem_cls = torch.randint(
            low=0, high=num_classes, size=(capacity, )).to(device)
        if cfg:
            few_shot_dataset = cfg.DATASETS.TRAIN[0]
            if 'shot' in few_shot_dataset:
                self.few_shot_mode = True

                # meta = MetadataCatalog.get(few_shot_dataset)
                # self.novel_pos = meta.novel_classes
                # print(meta.thing_classes)
                # print(self.novel_pos)
                # assert 0

    @torch.no_grad()
    def get_mem(self, cls):
        return self.memory[cls]

    def forward(self, x, classes):
        self.update(x, classes)
        # if self.few_shot_mode:
        #     return self.memory[-5:]

        return self.memory

    @torch.no_grad()
    def update(self, instances, classes):
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()

        new_ins = torch.cat([instances, self.memory], dim=0)
        self.memory = new_ins[:self.cap]
        self.mem_cls = torch.cat([classes, self.mem_cls], dim=0)[:self.cap]

        # for cls in __unique:
        #     if cls == self.num_classes:
        #         continue
        #     candidates = self.memory[cls]
        #     index = torch.where(classes == cls)
        #     new_ins = torch.cat([instances[index[0]], candidates])
        #     self.memory[cls] = new_ins[:self.cap]
        #     del new_ins, candidates


class memory_bank_ot3(torch.nn.Module):
    def __init__(self, num_classes=60, capacity=32, input_dim=1024, device='cpu', mem=None, cfg=None):
        super(memory_bank_ot3, self).__init__()

        self.fixed_memory = mem
        self.memory = torch.zeros(
            num_classes, capacity, input_dim).to(device)

        self.cap = capacity
        self.num_classes = num_classes
        self.device = device

    @torch.no_grad()
    def get_mem(self, get_cls=5):
        collected_cls = torch.randint(0, self.num_classes, (get_cls,))
        self.mem_cls = collected_cls.view(-1, 1).repeat(1, self.cap).cuda()
        return self.memory[collected_cls]

    def forward(self, x, classes, get_cls=5):
        self.update(x, classes)
        return self.get_mem(get_cls)

    @torch.no_grad()
    def update(self, instances, classes):
        __unique = torch.unique(classes)
        __unique = __unique.detach().cpu().numpy().tolist()
        # noise = self.memory.mean([0, 1], keepdim=True)
        # alpha=0.1
        # print(noise.shape)
        for cls in __unique:
            if cls == self.num_classes:
                continue
            candidates = self.memory[cls]
            index = torch.where(classes == cls)
            new_ins = torch.cat([instances[index[0]], candidates])[:self.cap]
            # new_ins = (1-alpha)*new_ins + alpha*noise
            self.memory[cls] = new_ins
            del new_ins, candidates


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=None, smoothing=0.2, dim=-1, weight=None):
        """
        Taken from: https://stackoverflow.com/a/66773267
        if smoothing == 0, it's one-hot method if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        # self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        n_cls = pred.shape[1]
        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_cls - 1))
            true_dist.scatter_(
                1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class generator(torch.nn.Module):
    def __init__(self, input_size):
        super(generator, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                data = m.weight.data
                # + torch.empty(data.shape[0], data.shape[1]).normal_(mean=0, std=0.02)
                m.weight.data = torch.eye(data.shape[0], data.shape[1]) + torch.empty(
                    data.shape[0], data.shape[1]).normal_(mean=0, std=0.02)

        self.mean_layer = nn.Linear(input_size, 1, bias=True)
        self.std_layer = torch.nn.Sequential(
            nn.Linear(input_size, 1, bias=True),
            nn.ReLU(),
        )

        self.mlp = torch.nn.Sequential(
            nn.Linear(input_size*2, input_size, bias=False),
            nn.ReLU(),
            nn.Linear(input_size, input_size, bias=False),
            nn.ReLU(),
        )
        for m in self.mlp:
            init_weights(m)

    def forward(self, x, label, repeat_time=16):
        C = x.shape[-1]
        mean = self.mean_layer(x).unsqueeze(1).repeat(1, repeat_time, C)
        std = self.std_layer(x).unsqueeze(1).repeat(1, repeat_time, C)
        label = label.unsqueeze(1).repeat(1, repeat_time).view(-1)

        z = torch.normal(mean=mean, std=std).view(-1, C)
        x = x.unsqueeze(1).repeat(1, repeat_time, 1).view(-1, C)
        x = self.mlp(torch.cat([x, z], dim=1))

        return x, label


class generator2(torch.nn.Module):
    def __init__(self, input_size):
        super(generator2, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                data = m.weight.data
                # + torch.empty(data.shape[0], data.shape[1]).normal_(mean=0, std=0.02)
                m.weight.data = torch.eye(data.shape[0], data.shape[1]) + torch.empty(
                    data.shape[0], data.shape[1]).normal_(mean=0, std=0.02)

        self.mean_layer = nn.Linear(input_size, input_size, bias=True)
        self.std_layer = torch.nn.Sequential(
            nn.Linear(input_size, input_size, bias=True),
            nn.ReLU(),
        )

        self.mlp = torch.nn.Sequential(
            nn.Linear(input_size*2, input_size, bias=False),
            nn.ReLU(),
            nn.Linear(input_size, input_size, bias=False),
            nn.ReLU(),
        )
        for m in self.mlp:
            init_weights(m)

    def forward(self, x, label, repeat_time=16):
        C = x.shape[-1]

        mean = self.mean_layer(x).repeat(repeat_time)
        std = self.std_layer(x).repeat(repeat_time)

        label = label.repeat(repeat_time)

        z = torch.normal(mean=mean, std=std)
        x = x.repeat(repeat_time, 1)
        x = self.mlp(torch.cat([x, z], dim=1))

        return x, label


class contrastive_loss(torch.nn.Module):
    def __init__(self, num_classes=2, capacity=32, input_dim=1024, device='cpu', tau=1):
        super(contrastive_loss, self).__init__()
        # self.criterion = LabelSmoothingCrossEntropy(epsilon=0)
        # self.criterion = LabelSmoothingLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        tau = 5e-3
        tau = 1e-2
        # tau = 1
        self.tau = tau
        self.norm = lambda a: torch.nn.functional.normalize(
            a, dim=-1)
        # self.norm = lambda a: a

    def forward(self, x, y, mem):

        # x shape: n, input_dim
        # y shape: n

        # mem shape                       : n_cls, cap, input_dim
        # -> mem1 shape (elements)        : (n_cls - 1) x cap, input_dim
        # -> mem2 shape (anchor - mean)   : n_cls, input_dim

        return self.call_loss3(x, y, mem, self.tau)
        # return self.call_loss(x, y, mem, self.tau)

    def call_loss(self, x, y, mem, tau):  # tau=5e-3

        mem_shape = mem.shape
        index = torch.arange(mem.shape[0])[None, :, None].repeat(
            x.shape[0], 1, mem_shape[1]).cuda()
        expand_y = y[:, None, None].repeat(1, mem_shape[0], mem_shape[1])
        expand_mem = mem[None, ...].repeat(x.shape[0], 1, 1, 1)

        l_pos = expand_mem[index == expand_y].view(
            x.shape[0], -1, x.shape[-1]).mean(dim=1)@x.T

        l_neg = expand_mem[index != expand_y].view(
            x.shape[0], -1, x.shape[-1])@x.T

        l_neg = l_neg.view(x.shape[0], -1)

        # k = 100
        # sorted, indices = torch.sort(l_neg, dim=-1, descending=True)
        # # indices = indices[:, :k]
        # # l_neg = torch.gather(l_neg,1, indices)
        # l_neg = sorted[:, :k]

        labels = torch.zeros(x.shape[0]).cuda().to(torch.long)
        labels[torch.arange(x.shape[0])] = 1

        logits = torch.cat([l_pos, l_neg], dim=1)

        return self.criterion(logits, labels)

    def call_loss2(self, x, y, mem, tau):  # tau=5e-3
        mem_shape = mem.shape
        index = torch.arange(mem.shape[0])[None, :, None].repeat(
            x.shape[0], 1, mem_shape[1]).cuda()
        expand_y = y[:, None, None].repeat(1, mem_shape[0], mem_shape[1])
        expand_mem = mem[None, ...].repeat(x.shape[0], 1, 1, 1)

        # normalize instance
        x_norm = self.norm(x)

        l_pos = self.norm(expand_mem[index == expand_y].view(
            x.shape[0], -1, x.shape[-1]).mean(dim=1)) @ x_norm.T / tau

        l_neg = self.norm(expand_mem[index != expand_y].view(
            x.shape[0], -1, x.shape[-1])) @ x_norm.T / tau

        k = 20
        # l_neg, _ = torch.topk(l_neg, k, dim=1)

        l_neg = l_neg.view(x.shape[0], -1)
        l_neg_high, _ = torch.topk(l_neg, k, dim=1)
        l_neg_low, _ = torch.topk(l_neg, k, dim=1, largest=False)
        l_neg = torch.cat([l_neg_high, l_neg_low], dim=1)

        labels = torch.zeros(x.shape[0]).cuda().to(torch.long)
        # labels[torch.arange(x.shape[0])] = 1
        l_pos = torch.gather(
            l_pos, dim=1, index=torch.arange(x.shape[0]).cuda()[..., None])

        logits = torch.cat([l_pos, l_neg], dim=1)
        print('pre-softmax :', logits)
        print('softmax :', torch.nn.functional.softmax(logits, dim=1)[0])

        return self.criterion(logits, labels)

    def call_loss3(self, x, y, mem, tau):  # tau=5e-3
        mem_shape = mem.shape
        label_mem = torch.arange(mem.shape[0]).cuda()
        index = label_mem[None, ...].repeat(
            x.shape[0], mem_shape[1], 1)

        expand_y = y[:, None, None].repeat(1, mem.shape[1], mem.shape[0])
        # expand_mem = mem[None, ...].repeat(x.shape[0], 1, 1, 1)

        # normalize instance
        x_norm = self.norm(x)

        # l_pos = self.norm(expand_mem[index == expand_y].view(
        #     x.shape[0], -1, x.shape[-1]).mean(dim=1)) @ x_norm.T / tau

        l_pos = torch.einsum('nd,bcd->ncb', x_norm,
                             self.norm(mem.mean(dim=1, keepdim=True))) / tau
        l_pos = l_pos.squeeze(1)[(y[..., None].repeat(
            1, mem.shape[0]) == label_mem[None, ...].repeat(y.shape[0], 1))]
        l_pos = l_pos.unsqueeze(1)
        # l_pos = l_pos.squeeze(1)
        # print(l_pos.shape)
        # l_neg = self.norm(expand_mem[index != expand_y].view(
        #     x.shape[0], -1, x.shape[-1])) @ x_norm.T / tau

        l_neg = torch.einsum('nd,bcd->ncb', x_norm, self.norm(mem)) / tau
        l_neg = l_neg[index != expand_y].view(x.shape[0], -1)

        # print(l_neg.shape)
        k = 40  # l_neg.shape[0]  # 20  # for mem update

        # l_neg_high, _ = torch.topk(l_neg, k, dim=1)
        # l_neg_low, _ = torch.topk(l_neg, k, dim=1, largest=False)
        # l_neg = torch.cat([l_neg_high, l_neg_low], dim=1)

        l_neg, _ = torch.topk(l_neg, k, dim=1, largest=True)

        labels = torch.zeros(x.shape[0]).to(torch.long).cuda()
        # labels[torch.arange(x.shape[0])] = 1

        logits = torch.cat([l_pos, l_neg], dim=1)

        # print('pre-softmax :', logits)
        # print('softmax :', torch.nn.functional.softmax(logits, dim=1)[0])

        return self.criterion(logits, labels)


def augment_bbox(bbox, image_shape, scale_factor=2):
    # print(bbox)
    x, y, x2, y2 = bbox
    w = x2 - x
    h = y2 - y

    x_center = x + w/2
    y_center = y + h/2

    scaled_w = w*scale_factor
    scaled_h = h*scale_factor

    new_bbox = [
        [x_center, y_center, scaled_w, scaled_h],
        [x_center, y_center, w, scaled_h],
        [x_center, y_center, scaled_w, h],
    ]

    new_bbox = [covert_center2_xywh(bb, image_shape) for bb in new_bbox]
    return new_bbox


def covert_center2_xywh(bbox, image_shape):
    width, height = image_shape

    x_c, y_c, w, h = bbox
    x = max(0, x_c - w/2)
    y = max(0, y_c - h/2)

    x2 = min(width, x+w)
    y2 = min(height, y+h)
    return [x, y, x2, y2]


def pair_L2_distance(feature1, feature2):
    # feature1: N, D
    # feature1: M, D

    feature1.unsqueeze(0).transpose(0, 1) - feature2.unsqueeze(0)

    dist = (feature1.unsqueeze(0).transpose(0, 1) - feature2.unsqueeze(0))**2

    dist = torch.nn.functional.relu(dist.sum(-1)).sqrt().cuda()
    return dist

# class loss()

#
# --- testing ---
#


if __name__ == '__main__':
    from torch.autograd import gradcheck
    from torch.nn.functional import normalize

    torch.manual_seed(0)

    M = torch.randn((3, 5, 7), dtype=torch.double, requires_grad=True)
    f = OptimalTransportFcn().apply

    print(torch.all(torch.isclose(sinkhorn(M), f(M))))
    print(torch.all(torch.isclose(sinkhorn(M), sinkhorn(
        torch.exp(-1.0 * M), logspace=True))))

    test = gradcheck(f, (M, None, None, 1.0, 1.0e-6, 1000,
                     False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (torch.exp(-1.0 * M), None, None, 1.0, 1.0e-6, 1000, True, 'block'), eps=1e-6, atol=1e-3,
                     rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 1.0, 1.0e-6, 1000,
                     False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000,
                     False, 'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, None, 10.0, 1.0e-6, 1000,
                     False, 'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    r = normalize(torch.rand(
        (M.shape[0], M.shape[1]), dtype=torch.double, requires_grad=False), p=1.0)
    c = normalize(torch.rand(
        (M.shape[0], M.shape[2]), dtype=torch.double, requires_grad=False), p=1.0)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-9, 1000, False,
                     'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    # with r and c inputs
    r = normalize(torch.rand(
        (M.shape[0], M.shape[1]), dtype=torch.double, requires_grad=True), p=1.0)
    c = normalize(torch.rand(
        (M.shape[0], M.shape[2]), dtype=torch.double, requires_grad=True), p=1.0)

    test = gradcheck(f, (M, r, None, 1.0, 1.0e-6, 1000, False,
                     'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, None, c, 1.0, 1.0e-6, 1000, False,
                     'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False,
                     'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 10.0, 1.0e-6, 1000, False,
                     'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    # shared r and c
    r = normalize(torch.rand(
        (1, M.shape[1]), dtype=torch.double, requires_grad=True), p=1.0)
    c = normalize(torch.rand(
        (1, M.shape[2]), dtype=torch.double, requires_grad=True), p=1.0)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False,
                     'block'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)

    test = gradcheck(f, (M, r, c, 1.0, 1.0e-6, 1000, False,
                     'full'), eps=1e-6, atol=1e-3, rtol=1e-6)
    print(test)


def Deconv_loss(feat):
    # C = torch.cov(feat)
    return torch.tensor([0.0]).cuda()
    C = ((feat-feat.mean())@(feat-feat.mean()).T) / \
        (feat.shape[1] + feat.shape[0])
    return 0.5*(torch.norm(C)**2 - torch.norm(torch.diag(C))**2)  # * 1e-20

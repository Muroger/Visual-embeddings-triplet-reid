import torch
import torch.nn as nn
import torch.nn.functional as F

import logger as log
import numpy as np

choices = ["BatchHard", "BatchSoft", "BatchHardWithSoftmax", "BatchHardSingleWithSoftmax", "BatchHardWithJunkSigmoid", "BatchHardWithJunkSoftmax"]

def calc_cdist(a, b, metric='euclidean'):
    diff = torch.unsqueeze(a, dim=1) - torch.unsqueeze(b, dim=0)#a[:, None, :] - b[None, :, :]
    if metric == 'euclidean':
        return torch.sqrt(torch.sum(torch.square(diff), dim=-1) + 1e-12)
    elif metric == 'sqeuclidean':
        return torch.sum(torch.square(diff), dim=-1)
    elif metric == 'cityblock':
        return torch.sum(diff.abs(), dim=-1)
    else:
        raise NotImplementedError("Metric %s has not been implemented!" % metric)


def _apply_margin(x, m):
    if isinstance(m, float):
        return (x + m).clamp(min=0)
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % m)


def batch_hard(cdist, pids, margin):
    """Computes the batch hard loss as in arxiv.org/abs/1703.07737.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
    """
    mask_pos = (pids[None, :] == pids[:, None]).float()

    ALMOST_INF = 9999.9
    furthest_positive = torch.max(cdist * mask_pos, dim=0)[0]
    furthest_negative = torch.min(cdist + ALMOST_INF*mask_pos, dim=0)[0]
    #furthest_negative = torch.stack([
    #    torch.min(row_d[row_m]) for row_d, row_m in zip(cdist, mask_neg)
    #]).squeeze() # stacking adds an extra dimension

    return _apply_margin(furthest_positive - furthest_negative, margin)


class BatchHard(nn.Module):
    def __init__(self, m, **kwargs):
        super(BatchHard, self).__init__()
        self.name = "BatchHard(m={})".format(m)
        self.m = m

    def forward(self, dist, pids, **kwargs):
        return batch_hard(dist, pids, self.m)


def batch_soft(cdist, pids, margin, T=1.0):
    """Calculates the batch soft.
    Instead of picking the hardest example through argmax or argmin,
    a softmax (softmin) is used to sample and use less difficult examples as well.

    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
        T (float): The temperature of the softmax operation.
    """
    # mask where all positivies are set to true
    mask_pos = pids[None, :] == pids[:, None]
    mask_neg = 1 - mask_pos.data

    # only one copy
    cdist_max = cdist.clone()
    cdist_max[mask_neg] = -float('inf')
    cdist_min = cdist.clone()
    cdist_min[mask_pos] = float('inf')

    # NOTE: We could even take multiple ones by increasing num_samples,
    #       the following `gather` call does the right thing!
    idx_pos = torch.multinomial(F.softmax(cdist_max/T, dim=1), num_samples=1)
    idx_neg = torch.multinomial(F.softmin(cdist_min/T, dim=1), num_samples=1)
    positive = cdist.gather(dim=1, index=idx_pos)[:,0]  # Drop the extra (samples) dim
    negative = cdist.gather(dim=1, index=idx_neg)[:,0]

    return _apply_margin(positive - negative, margin)


class BatchSoft(nn.Module):
    """BatchSoft implementation using softmax.
    
    Also by Tristani as Adaptivei Weighted Triplet Loss.
    """

    def __init__(self, m, T=1.0, **kwargs):
        """
        Args:
            m: margin
            T: Softmax temperature
        """
        super(BatchSoft, self).__init__()
        self.name = "BatchSoft(m={}, T={})".format(m, T)
        self.m = m
        self.T = T

    def forward(self, dist, pids):
        return batch_soft(dist, pids, self.m, self.T)


class BatchHardWithJunkSigmoid(nn.Module):
    def __init__(self, m, num_junk_images, **kwargs):
        super().__init__()
        self.name = "BatchHardWithSigmoid(m={}, J={})".format(m, num_junk_images)
        self.batch_hard = BatchHard(m)
        self.cross_entropy = nn.BCEWithLogitsLoss(reduce=False)
        self.num_junk_images = num_junk_images
        self.sigmoid = nn.Sigmoid()

    def forward(self, pids, endpoints, **kwargs):
        # only one triplet embedding is passed
        triplet = endpoints["triplet"][0]
        triplet_pids = pids
        dist = calc_cdist(triplet, triplet)
        bh_loss = self.batch_hard(dist, triplet_pids)
        # here is no data parallel anymore
        targets = torch.zeros_like(pids, dtype=torch.float).unsqueeze(1)
        targets[-self.num_junk_images:] = 1.0
        ce_loss = self.cross_entropy(endpoints["junk"], targets)
        ce_loss_f = float(var2num(torch.mean(ce_loss)))
        bh_loss_f = float(var2num(torch.mean(bh_loss)))
        acc = self._calc_junk_acc(endpoints["junk"], targets)
        print("bh loss {:.3f} ce loss: {:.3f} acc: {:.3f}".format(bh_loss_f, ce_loss_f, acc))
        log.write("loss", (ce_loss_f, bh_loss_f), dtype=np.float32)
        return torch.mean(bh_loss) + torch.mean(ce_loss)

    def _calc_junk_acc(self, logits, targets, threshold=0.5):
        probs = self.sigmoid(logits)
        predicted = (probs > threshold).float()
        return torch.sum(targets == predicted).float() / targets.shape[0]
        
class BatchHardWithJunkSoftmax(nn.Module):
    def __init__(self, m, num_junk_images, **kwargs):
        super().__init__()
        self.name = "BatchHardWithJunkSoftmax(m={}, J={})".format(m, num_junk_images)
        self.batch_hard = BatchHard(m)
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.num_junk_images = num_junk_images
        self.softmax = nn.Softmax()

    def forward(self, pids, endpoints, **kwargs):
        triplet = endpoints["triplet"][0][:-self.num_junk_images]
        triplet_pids = pids[:-self.num_junk_images]
        dist = calc_cdist(triplet, triplet)
        bh_loss = self.batch_hard(dist, triplet_pids)
        # here is no data parallel anymore
        #class 0  no junk, class 1 junk
        targets = torch.zeros_like(pids, dtype=torch.long)
        targets[-self.num_junk_images:] = 1
        ce_loss = self.cross_entropy(endpoints["junk"], targets)
        ce_loss_f = float(var2num(torch.mean(ce_loss)))
        bh_loss_f = float(var2num(torch.mean(bh_loss)))
        acc = self._calc_junk_acc(endpoints["junk"], targets)
        print("bh loss {:.3f} ce loss: {:.3f} acc: {:.3f}".format(bh_loss_f, ce_loss_f, acc))
        log.write("loss", (ce_loss_f, bh_loss_f), dtype=np.float32)

        return torch.mean(bh_loss) + torch.mean(ce_loss)

    def _calc_junk_acc(self, logits, targets, threshold=0.5):
        predicted = torch.max(logits, dim=1)
        predicted = predicted[1]
        return torch.sum(targets == predicted).float() / targets.shape[0]

class BatchHardWithSoftmax(nn.Module):
    """SoftmaxLoss or Softmax Classifier uses the NegativeLogLikelyLoss
    and the softmax function.
    The torch implementation of CrossEntropy includes the softmax.

    """

    def __init__(self, m, a=1.0, **kwargs):
        super().__init__()
        self.batch_hard = BatchHard(m)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.name = "BatchHardWithSoftmax(m={}, a={})".format(m, a)
        self.a = a

    def forward(self, dist, pids, endpoints, **kwargs):
        batch_hard_loss = self.batch_hard(dist, pids)
        if self.a > 0:
            cross_entropy_loss = 0.0
            for softmax in endpoints["soft"]:
                cross_entropy_loss += self.cross_entropy(softmax, pids)
                ce_loss = float(var2num(cross_entropy_loss))
            ce_loss = float(var2num(cross_entropy_loss))
            bh_loss = float(var2num(torch.mean(batch_hard_loss)))
            print("bh loss {:.3f} ce loss: {:.3f}".format(bh_loss, ce_loss))
            log.write("loss", (bh_loss, ce_loss), dtype=np.float32)
            return batch_hard_loss + self.a * cross_entropy_loss
        else:
            return batch_hard_loss


class BatchHardSingleWithSoftmax(nn.Module):
    """SoftmaxLoss or Softmax Classifier uses the NegativeLogLikelyLoss
    and the softmax function.
    The torch implementation of CrossEntropy includes the softmax.

    """

    def __init__(self, m, **kwargs):
        super().__init__()
        self.batch_hard = BatchHard(m)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.name = "BatchHardSingleWithSoftmax(m={})".format(m)

    def forward(self, pids, endpoints, **kwargs):
        batch_hard_loss = 0.0
        bh_losses = []

        for triplet in endpoints["triplet"]:
            dist = calc_cdist(triplet, triplet)
            bh = self.batch_hard(dist, pids)
            batch_hard_loss += bh
            f_bh = float(var2num(torch.mean(bh)))
            bh_losses.append(f_bh)

        bh_loss_overall = float(var2num(torch.mean(batch_hard_loss)))

        cross_entropy_loss = 0.0
        ce_losses = []
        for softmax in endpoints["soft"]:
            ce = self.cross_entropy(softmax, pids)
            cross_entropy_loss += ce
            f_ce = float(var2num(ce))
            ce_losses.append(f_ce)
        
        ce_loss_overall = float(var2num(cross_entropy_loss))
        
        print("bh loss {:.3f} ce loss: {:.3f}".format(bh_loss_overall, ce_loss_overall))
        loss_info = [bh_loss_overall] + bh_losses + [ce_loss_overall] + ce_losses
        log.write("loss", loss_info, dtype=np.float32)
        return batch_hard_loss + cross_entropy_loss

def var2num(x):
    return x.data.cpu().numpy()


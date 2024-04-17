# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions.beta import Beta
from rdcd.models.model import get_encoder
from rdcd.lib.util import call_using_args

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, args, base_encoder, K=65536, m=0.99, T=0.07, pmf='beta', pos_alpha=None, pos_beta=None, neg_alpha=None, \
                 c=0.6, train_x=False, train_c=False, decoupled=False, theta=None, post_norm=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.pmf = pmf
        self.pos_alpha = pos_alpha
        self.pos_beta = pos_beta
        self.neg_alpha = neg_alpha

        self.eps = 1e-5
        self.decoupled = decoupled
        self.theta = theta
        self.post_norm = post_norm

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = get_encoder(args, base_encoder)
        self.encoder_k = get_encoder(args, base_encoder)


        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.koleo_loss = KoLeoLoss()
        
        # create the queue
        self.register_buffer("queue", torch.randn(args.dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def decouple_norm_from_direction(self, v):
        v_hat = nn.functional.normalize(v, dim=1).clone().detach()
        v_norm = v.norm(2, dim=1, keepdim=True)
        return v_hat*v_norm

    def forward(self, im_q, im_k=None, eval=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        feat_q, q = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)
        if self.theta:
            if self.decoupled:
                q = self.decouple_norm_from_direction(q)
            q = self.tp(q)

        if eval:
            #return feat_q
            return q


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            feat_k, k = self.encoder_k(im_k)  # keys: NxC

            # k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            
        queue = self.queue.clone().detach()

        # # for positives pairs
        # if self.pos_alpha:
        #     if self.pmf == 'beta':
        #         distribution = Beta(self.pos_alpha, self.pos_beta)
        #         pos_lambdas = distribution.sample(q.shape).cuda()
        #     elif self.pmf == 'uniform':
        #         pos_lambdas = torch.FloatTensor(np.random.rand(q.shape)).cuda()
        #     else:
        #         pos_lambdas = torch.FloatTensor(np.random.rand(q.shape[0])).cuda()
        #         pos_lambdas = pos_lambdas.view(q.shape[0], 1)

        #     pos_lambdas += 1
        #     q_mix = pos_lambdas * q + (1 - pos_lambdas) * k
        #     k_mix = pos_lambdas * k + (1 - pos_lambdas) * q
        #     q, k = q_mix, k_mix

        # if self.post_norm:
        #     q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)


        # # # for negatives pairs
        # if self.neg_alpha:
        #     if self.pmf == 'beta':
        #         distribution = Beta(self.neg_alpha, self.neg_alpha)
        #         neg_lambdas = distribution.sample(queue.shape).cuda()
        #     elif self.pmf == 'uniform':
        #         neg_lambdas = torch.FloatTensor(np.random.rand(queue.shape)).cuda()
        #     else:
        #         neg_lambdas = torch.FloatTensor(np.random.rand(queue.shape[0])).cuda()
        #         neg_lambdas = neg_lambdas.view(queue.shape[0], 1)

        #     indices = torch.randperm(queue.shape[0]).cuda()
        #     queue = neg_lambdas * queue + (1 - neg_lambdas) * queue[indices]

        #     if self.post_norm:
        #         queue = F.normalize(queue, dim=1)

        if not self.theta:
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) # euc
            l_neg = torch.einsum("nc,ck->nk", [q, queue]) # euc
        else:
            k = self.tp(k)
            l_pos = -self.hyperbolic_dis(q, k) # hyp
            l_neg = -self.hyperbolic_dis(q.unsqueeze(1), queue.T.unsqueeze(0)).squeeze()


        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        moco_loss = self.loss(logits, labels)
        max_non_match_sim = torch.max(l_neg, dim=1)[0] # shape (256, 65536) -> (256,)
        hardneg_loss = -(1-max_non_match_sim).clamp(min=1e-3).log().mean()
        #print(hardneg_loss)
        loss = moco_loss + hardneg_loss * 5
        #loss = moco_loss
        stats = {
                    "positive_sim": l_pos.mean(),
                    "negative_sim": l_neg.mean(),
                    "nearest_negative_sim": torch.mean(torch.max(l_neg, dim=1)[0]),
                    "center_l2_norm": q.mean(dim=0).pow(2).sum().sqrt(),
                    "Moco loss": moco_loss,
                    "Hardneg loss": hardneg_loss*5,
                }
        
        return feat_q, q, loss, stats

    # def forward(self, im_q, im_k=None, eval=False):
    #     """
    #     Input:
    #         im_q: a batch of query images
    #         im_k: a batch of key images
    #     Output:
    #         logits, targetss
    #     """

    #     # compute query features
    #     q = self.encoder_q(im_q)  # queries: NxC
    #     euc_q = q.clone()
    #     # q = nn.functional.normalize(q, dim=1)
    #     if self.theta:
    #         if self.decoupled:
    #             q = self.decouple_norm_from_direction(q)
    #         q = self.tp(q)

    #     if eval:
    #         return q

    #     # compute key features
    #     with torch.no_grad():  # no gradient to keys
    #         self._momentum_update_key_encoder()  # update the key encoder

    #         # shuffle for making use of BN
    #         im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

    #         k = self.encoder_k(im_k)  # keys: NxC
    #         # k = nn.functional.normalize(k, dim=1)
    #         k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            
    #     queue = self.queue.clone().detach()

    #     # # for positives pairs
    #     # if self.pos_alpha:
    #     #     if self.pmf == 'beta':
    #     #         distribution = Beta(self.pos_alpha, self.pos_beta)
    #     #         pos_lambdas = distribution.sample(q.shape).cuda()
    #     #     elif self.pmf == 'uniform':
    #     #         pos_lambdas = torch.FloatTensor(np.random.rand(q.shape)).cuda()
    #     #     else:
    #     #         pos_lambdas = torch.FloatTensor(np.random.rand(q.shape[0])).cuda()
    #     #         pos_lambdas = pos_lambdas.view(q.shape[0], 1)

    #     #     pos_lambdas += 1
    #     #     q_mix = pos_lambdas * q + (1 - pos_lambdas) * k
    #     #     k_mix = pos_lambdas * k + (1 - pos_lambdas) * q
    #     #     q, k = q_mix, k_mix

    #     # if self.post_norm:
    #     #     q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)


    #     # # # for negatives pairs
    #     # if self.neg_alpha:
    #     #     if self.pmf == 'beta':
    #     #         distribution = Beta(self.neg_alpha, self.neg_alpha)
    #     #         neg_lambdas = distribution.sample(queue.shape).cuda()
    #     #     elif self.pmf == 'uniform':
    #     #         neg_lambdas = torch.FloatTensor(np.random.rand(queue.shape)).cuda()
    #     #     else:
    #     #         neg_lambdas = torch.FloatTensor(np.random.rand(queue.shape[0])).cuda()
    #     #         neg_lambdas = neg_lambdas.view(queue.shape[0], 1)

    #     #     indices = torch.randperm(queue.shape[0]).cuda()
    #     #     queue = neg_lambdas * queue + (1 - neg_lambdas) * queue[indices]

    #     #     if self.post_norm:
    #     #         queue = F.normalize(queue, dim=1)

    #     if not self.theta:
    #         l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) # euc
    #         l_neg = torch.einsum("nc,ck->nk", [q, queue]) # euc
    #     else:
    #         k = self.tp(k)
    #         l_pos = -self.hyperbolic_dis(q, k) # hyp
    #         l_neg = -self.hyperbolic_dis(q.unsqueeze(1), queue.T.unsqueeze(0)).squeeze()


    #     # logits: Nx(1+K)
    #     logits = torch.cat([l_pos, l_neg], dim=1)

    #     # apply temperature
    #     logits /= self.T

    #     # labels: positive key indicators
    #     labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

    #     # dequeue and enqueue
    #     self._dequeue_and_enqueue(k)

    #     moco_loss = self.loss(logits, labels)
    #     #koleo_loss = 30*self.koleo_loss(logits)
    #     #koleo_loss = 10*self.koleo_loss(logits)
    #     #loss = moco_loss + koleo_loss
    #     loss = moco_loss
    #     #moco_loss = self.loss(logits, labels)
    #     #max_non_match_sim = torch.max(l_neg, dim=1)[0] # shape (256, 65536) -> (256,)
    #     #hardneg_loss = -(1-max_non_match_sim).clamp(min=1e-3).log().mean()
    #     #print(hardneg_loss)
    #     #loss = moco_loss + hardneg_loss * 3
    #     stats = {
    #                 "positive_sim": l_pos.mean(),
    #                 "negative_sim": l_neg.mean(),
    #                 "nearest_negative_sim": torch.mean(torch.max(l_neg, dim=1)[0]),
    #                 "center_l2_norm": q.mean(dim=0).pow(2).sum().sqrt(),
    #                 "Moco loss": moco_loss,
    #             }
        
    #     return euc_q, q, loss, stats

    def loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(logits, labels)
        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss
import torch
import torch.nn as nn
def si_snr_loss_fn(output, target):
    """
    calculate si_snr loss on output audio and target audio
    output: [B, T'], target: [B, T]. the T' will be modified to be equal to T
    Returns tensor with containing si_snr loss
    """
    def si_snr(s1, s2, eps=1e-8):
        def l2_norm(s1, s2):
            return torch.sum(s1 *s2, -1, keepdim= True)
        s1_s2_norm = l2_norm(s1, s2)
        s2_s2_norm = l2_norm(s2, s2)
        s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
        e_nosie = s1 - s_target
        target_norm = l2_norm(s_target, s_target)
        noise_norm = l2_norm(e_nosie, e_nosie)
        snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
        return torch.mean(snr)
    ## shrink the audio
    if output.size(1) == target.size(1):
        pass
    elif output.size(1) > target.size(1):
        output= output[:,:target.size(1)]
    else:
        dim = target.size(1) - output.size(1)
        const_tensor = torch.zeros(target.size(0), dim).to(output.device)
        output = torch.cat((output,const_tensor), dim = 1)
        pass
    return -1 * si_snr(output, target)


def vec_dot_mul(y1, y2):
    dot_mul = torch.sum(torch.mul(y1, y2), dim=-1)
    # print('dot', dot_mul.size())
    return dot_mul

def vec_normal(y):
    normal_ = torch.sqrt(torch.sum(y**2, dim=-1))
    # print('norm',normal_.size())
    return normal_

def cos_loss_fn(est, ref): # -cos
    '''
    est, ref: [batch, ..., n_sample]
    '''
    # print(est.size(), ref.size(), flush=True)
    cos_sim = - torch.div(vec_dot_mul(est, ref), # [batch, ...]
        torch.mul(vec_normal(est), vec_normal(ref)))
    loss = torch.mean(cos_sim)
    return loss


mseLoss = torch.nn.MSELoss()
def mse_loss_fn(est,ref):
    return mseLoss(est, ref)

def rmse_loss_fn(y1, y2, RL_epsilon=1e-8, index_=2.0):
   # y1, y2 : [N, F, T] or [N, c, F, T]
    relative_loss = torch.abs(y1-y2)/(torch.abs(y1)+torch.abs(y2)+RL_epsilon)
    loss = torch.pow(relative_loss, index_)
    loss = torch.mean(torch.sum(loss, -2))
    return loss

def rmse_sisnr_loss_fn(y1,y2, output, target, ratio = 1):
    """
    combine rmse with sisnr loss
    """
    rmse_loss = rmse_loss_fn(y1, y2)
    si_snr_loss = si_snr_loss_fn(output,target)
    return rmse_loss + ratio * si_snr_loss

crossEntropyLoss = nn.CrossEntropyLoss()
def cross_entropy_loss_fn(src, tgt):
    """
    src with shape (B, n_q, T, K)
    tgt with shape (n_q, B, T)
    """
    src_ = src.permute(0, 3, 2, 1) # [B, K , T, n_q]
    tgt_ = tgt.permute(1, 2, 0) # [B, T, n_q]
    return crossEntropyLoss(src_, tgt_)

kl_loss = torch.nn.KLDivLoss(reduce="batchmean")

def kl_div_loss_fn(src, tgt):
    """
    src : [B, T, C]
    tgt : [B, T]
    """
    src_ = torch.log_softmax(src, dim = -1)
    tgt_ = torch.nn.functional.one_hot(tgt, src.shape[-1]).to(src.dtype) # [B, T, C]
    return kl_loss(src_, tgt_)

def selm_loss_fn():
    """
    return kl div loss as well as the mse loss

    """
    return kl_div_loss_fn, mse_loss_fn
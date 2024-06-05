import torch
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
        const_tensor = torch.zeros(target.size(0), dim)
        output = torch.cat((output,const_tensor), dim = 1)
        pass
    return si_snr(output, target)
import torch
import torch.nn.functional as F


# The code was inspired from https://github.com/naoto0804/pytorch-AdaIN.
def calc_mean_std(feat, eps=1e-5, mask=None):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    if mask is not None:
        msize = mask.size()
        assert (msize[0] == size[0] and msize[2] == size[2] and msize[3] == size[3])
    N, C = size[:2]
    if mask is not None:
        cnt = mask.view(N, 1, -1).sum(2)
        mf = mask * feat
        mf = mf.view(N, C, -1).sum(2)
        mf /= cnt
        feat_mean = mf.view(N, C, 1, 1)
        vf = ((feat - feat_mean) ** 2) * mask
        vf = vf.view(N, C, -1).sum(2)
        vf = vf / (cnt - 1)
        vf += eps
        feat_std = vf.sqrt().view(N, C, 1, 1)
    else:
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat, c_mask=None, s_mask=None):
    assert (content_feat.size()[:2] == style_feat.size()[:2])

    size = content_feat.size()
    H, W = size[2], size[3]
    msk = None
    if c_mask is not None:
        msk = F.interpolate(c_mask, (H, W))

    s_size = style_feat.size()
    s_H, s_W = s_size[2], s_size[3]
    s_msk = None
    if s_mask is not None:
        s_msk = F.interpolate(s_mask, (s_H, s_W))

    style_mean, style_std = calc_mean_std(style_feat, mask=s_msk)
    content_mean, content_std = calc_mean_std(content_feat, mask=msk)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)

    if msk is not None:
        return (normalized_feat * style_std.expand(size) + style_mean.expand(size)) * msk + content_feat * (1 - msk)
    else:
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

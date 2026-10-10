import torch
from functools import partial
import torch.nn.functional as F

def modify_grad(x, inds, factor = 0.):
    inds = inds.expand_as(x)
    x[inds] *= factor

    return x

def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor

    return x

def global_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        #a_ and b_ shapes are (batch size, embedding, patch, patch) 
        #embeding:768
        with torch.no_grad():
            #point dist shape is (batch size, 1, patch, patch)
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)

        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()

        #print("point dist:", point_dist.reshape(-1).shape, point_dist.numel() * (1 - p), "p", p)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        #print("thresh shape:", thresh.shape, thresh)
        #exit()

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1), b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss

def global_cosine_focal(a, b, p=0.9, alpha=2., min_grad=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()

        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh

        focal_factor = focal_factor ** alpha
        focal_factor = torch.clip(focal_factor, min=min_grad)

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1), b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=focal_factor)
        b_.register_hook(partial_func)
    
    return loss

def global_cosine_hm(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    weight = [1, 1, 1]
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.view(a_.shape[0], -1), b_.view(b_.shape[0], -1))) * weight[item]
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    return loss

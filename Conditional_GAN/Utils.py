import torch
import torch.nn as nn

def gradient_penalty(gen_image, real_image, critic, labels, device = 'cpu'):

    [batch_size, Num_Ch, H, W] = real_image.shape
    Epsilon = torch.rand(batch_size, 1, 1, 1).repeat(1, Num_Ch, H, W).to(device)
    Interp_Image = Epsilon * real_image + (1 - Epsilon) * gen_image
    critic_Inter = critic(Interp_Image, labels)

    grad_respect_interp = torch.autograd.grad(
        inputs=Interp_Image,
        outputs=critic_Inter,
        grad_outputs=torch.ones_like(critic_Inter),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_respect_interp = grad_respect_interp.view(grad_respect_interp.shape[0], -1)
    L2Norm_Grad = grad_respect_interp.norm(2, dim=1)

    grad_penalty_final = torch.mean((L2Norm_Grad - 1)**2)

    return grad_penalty_final
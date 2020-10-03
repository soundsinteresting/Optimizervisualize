import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig, clf, scatter, legend, title, figure
import numpy as np


def decay_lr(opt, max_iter, start_iter, initial_lr):
    """Decay learning rate linearly till 0."""
    coeff = -initial_lr / (max_iter - start_iter)
    for pg in opt.param_groups:
        pg['lr'] += coeff

def getGradNorm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        try:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        except:
            print(name)
    total_norm = total_norm ** (1. / 2)
    return total_norm

def plot_losses(g_losses, d_losses, grad_normG, grad_normD, dir_path):
    plot(g_losses)
    title('g loss')
    savefig(dir_path + "/g_losses.png")
    clf()
    plot(d_losses)
    title('d loss')
    savefig(dir_path + "/d_losses.png")
    clf()
    plot(grad_normG)
    title('G norm (square root of sum G norm)')
    savefig(dir_path + "/grad_normG.png")
    clf()
    plot(grad_normD)
    title('D norm (square root of sum D norm)')
    savefig(dir_path + "/grad_normD.png")
    clf()
    np.save(dir_path + 'Dloss.npy', d_losses)
    np.save(dir_path + 'Gloss.npy', g_losses)
    np.save(dir_path + 'Dgram.npy', grad_normD)
    np.save(dir_path + 'Ggram.npy', grad_normG)

def saveproj(y, y_hat, i ,dir_path):
    fig=figure(figsize=(5, 5))
    scatter(y_hat.data.numpy(), np.zeros(y_hat.shape[0]), label='Fake', s=100)
    scatter(y.data.numpy(), np.zeros(y.shape[0]), label='Real', s=50)
    title('Disc_projection')
    legend()
    # plt.savefig(save_path + '/dprojection_epoch%05d.jpg' % (i/plot_freq))
    savefig(dir_path + '/dprojection_epoch%05d.pdf'%(i), bbox_inches='tight')
    clf()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

prefix = 'test'

import os
if not os.path.exists(prefix):
    os.makedirs(prefix)

# Define modal

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100 + 10, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024 + 10, 128 * 8 * 8)
        self.bn2 = nn.BatchNorm1d(128 * 8 * 8)
        self.cvt1 = nn.ConvTranspose2d(128 + 10, 64, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.cvt2 = nn.ConvTranspose2d(64 + 10, 3, 4, 2, 1)
    def forward(self, z, label):
        x = F.relu(self.bn1(self.fc1(torch.cat([z, label], 1))))
        x = F.relu(self.bn2(self.fc2(torch.cat([x, label], 1))))
        label = label.view(-1, 10, 1, 1)
        x = F.relu(self.bn3(self.cvt1(torch.cat([x.view(-1, 128, 8, 8), label.expand(x.size(0), 10, 8, 8)], 1))))
        return F.sigmoid(self.cvt2(torch.cat([x, label.expand(x.size(0), 10, 16, 16)], 1)))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cv1 = nn.Conv2d(3 + 10, 64, 4, 2, 1)
        self.cv2 = nn.Conv2d(64 + 10, 128, 4, 2, 1)
        self.bm1 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8 + 10, 1024)
        self.bm2 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024 + 10, 1)
    def forward(self, x, label):
        label_expand = label.view(-1, 10, 1, 1)
        x = F.leaky_relu(self.cv1(torch.cat([x, label_expand.expand(x.size(0), 10, 32, 32)], 1)))
        x = F.leaky_relu(self.bm1(self.cv2(torch.cat([x, label_expand.expand(x.size(0), 10, 16, 16)], 1))))
        x = F.leaky_relu(self.bm2(self.fc1(torch.cat([x.view(-1, 128 * 8 * 8), label], 1))))
        return F.sigmoid(self.fc2(torch.cat([x, label], 1)))

# Implement modal

import torch.optim as optim

lr = 1e-4
cuda = False

G, D = Generator(), Discriminator()
if cuda:
    G, D = G.cuda(), D.cuda()
G_optim, D_optim = optim.Adam(G.parameters(), lr = lr), optim.Adam(D.parameters(), lr = lr)

# Load data

from torchvision import datasets, transforms
cifar10 = datasets.CIFAR10('data', train=True, download=True, transform=transforms.ToTensor())
images = torch.stack([cifar10[i][0] for i in range(len(cifar10))])
orig_labels = torch.LongTensor([cifar10[i][1] for i in range(len(cifar10))])
labels = torch.zeros(images.size(0), 10).scatter_(1, orig_labels.view(-1, 1), 1)
if cuda:
    images, orig_labels, labels = images.cuda(), orig_labels.cuda(), labels.cuda()

# Load and save modal
def load_state():
    if os.path.exists(prefix + '/checkpoint/G.data'):
        G.load_state_dict(torch.load(prefix + '/checkpoint/G.data'))
    if os.path.exists(prefix + '/checkpoint/D.data'):
        D.load_state_dict(torch.load(prefix + '/checkpoint/D.data'))
    if os.path.exists(prefix + '/checkpoint/G_optim.data'):
        G_optim.load_state_dict(torch.load(prefix + '/checkpoint/G_optim.data'))
    if os.path.exists(prefix + '/checkpoint/D_optim.data'):
        D_optim.load_state_dict(torch.load(prefix + '/checkpoint/D_optim.data'))
    begin_epoch = 0
    if os.path.exists(prefix + '/checkpoint/epoch.data'):
        begin_epoch = torch.load(prefix + '/checkpoint/epoch.data')
    return begin_epoch

def save_state(epoch):
    if not os.path.exists(prefix + '/checkpoint'):
        os.makedirs(prefix + '/checkpoint')
    torch.save(G.state_dict(), prefix + '/checkpoint/G.data')
    torch.save(D.state_dict(), prefix + '/checkpoint/D.data')
    torch.save(G_optim.state_dict(), prefix + '/checkpoint/G_optim.data')
    torch.save(D_optim.state_dict(), prefix + '/checkpoint/D_optim.data')
    torch.save(epoch, prefix + '/checkpoint/epoch.data')

# Test

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plt_images(images, rows, cols):
    fig = plt.figure(figsize=(cols,rows))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    for i, x in enumerate(images[:cols * rows]):
        plt.subplot(rows, cols, i + 1)
        plt.axis('off')
        plt.imshow(np.rollaxis(x, 0, 3), norm=colors.NoNorm())
    return fig

plt_labels = Variable(torch.zeros(100, 10).scatter_(1, torch.LongTensor([[i]*10 for i in range(10)]).view(-1, 1), 1), volatile = True)
plt_z = Variable(torch.randn(100, 100), volatile = True)
if cuda:
    plt_labels, plt_z = plt_labels.cuda(), plt_z.cuda()

def plt_gen_with_random_noise(index):
    z = Variable(torch.randn(100, 100), volatile = True)
    if cuda:
        z = z.cuda()
    fig = plt_images(G(z, plt_labels).data.cpu().numpy(), 10, 10)
    if not os.path.exists(prefix + '/plt_random'):
        os.makedirs(prefix + '/plt_random')
    fig.savefig(prefix + '/plt_random/%s.png' % str(index).zfill(5))
    plt.close(fig)

def plt_gen_with_fixed_noise(index):
    fig = plt_images(G(plt_z, plt_labels).data.cpu().numpy(), 10, 10)
    if not os.path.exists(prefix + '/plt_fixed'):
        os.makedirs(prefix + '/plt_fixed')
    fig.savefig(prefix + '/plt_fixed/%s.png' % str(index).zfill(5))
    plt.close(fig)

from sklearn.manifold import TSNE
n_tsne_sample = 1000

def plt_tsne(index):
    indices = torch.from_numpy(np.random.randint(0, images.size(0), n_tsne_sample))
    if cuda:
        indices = indices.cuda()
    label, z = Variable(labels[indices], volatile = True), Variable(torch.randn(n_tsne_sample, 100), volatile = True)
    if cuda:
        z = z.cuda()
    source, target = images[indices].cpu().view(n_tsne_sample, 3 * 32 * 32).numpy(), G(z, label).data.cpu().view(n_tsne_sample, 3 * 32 * 32).numpy()
    model = TSNE(n_components=2)
    output = model.fit_transform(np.vstack([source, target]))
    source, target = output[:source.shape[0], :], output[source.shape[0]:, :]
    label = orig_labels[indices].cpu().numpy()

    fig = plt.figure(figsize=(10, 5))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)
    src = fig.add_subplot(1, 2, 1)
    tar = fig.add_subplot(1, 2, 2)
    src.scatter(source[:, 0], source[:, 1], s=15, c=label, cmap='plasma', edgecolors='none', marker='o')
    tar.scatter(target[:, 0], target[:, 1], s=15, c=label, cmap='plasma', edgecolors='none', marker='o')
    src.axis('off')
    tar.axis('off')
    if not os.path.exists(prefix + '/plt_tsne'):
        os.makedirs(prefix + '/plt_tsne')
    plt.savefig(prefix + '/plt_tsne/%s.png' % str(index).zfill(5))
    plt.close(fig)

def test(index):
    plt_gen_with_random_noise(index)
    plt_gen_with_fixed_noise(index)
    plt_tsne(index)

# Train

n_epoch = 20000
batch_size = 128
log_interval = 1
plot_interval = 100
save_interval = 100

def train():
    G.train()
    D.train()
    zeros = Variable(torch.zeros(batch_size, 1))
    ones = Variable(torch.ones(batch_size, 1))
    if cuda:
        zeros, ones = zeros.cuda(), ones.cuda()
    with open(prefix + '/log.txt', 'a') as log:
        for i in range(load_state(), n_epoch):
            for _ in range(5):
                G.zero_grad()
                D.zero_grad()
                indices = torch.from_numpy(np.random.randint(0, images.size(0), batch_size))
                if cuda:
                    indices = indices.cuda()
                realx, label, z = Variable(images[indices]), Variable(labels[indices]), Variable(torch.randn(batch_size, 100))
                if cuda:
                    z = z.cuda()
                fakex = G(z, label)
                realy, fakey = D(realx, label), D(fakex, label)
                D_loss = F.binary_cross_entropy(realy, ones) + F.binary_cross_entropy(fakey, zeros)
                D_loss.backward()
                D_optim.step()
            G.zero_grad()
            D.zero_grad()
            label = Variable(torch.zeros(batch_size, 10).scatter_(1, torch.from_numpy(np.random.randint(0, 10, batch_size)).view(-1, 1), 1))
            z = Variable(torch.randn(batch_size, 100))
            if cuda:
                label, z = label.cuda(), z.cuda()
            G_loss = F.binary_cross_entropy(D(G(z, label), label), ones)
            G_loss.backward()
            G_optim.step()
            if i % log_interval == 0:
                info = 'Epoch: %d; D_loss: %s; G_loss: %s' % (i, D_loss.data[0], G_loss.data[0])
                print(info)
                log.write(info + '\n')
            if i % plot_interval == 0:
                G.eval()
                D.eval()
                test(i // plot_interval)
                G.train()
                D.train()
            if i % save_interval == 0:
                save_state(i)

if __name__ == '__main__':
    train()

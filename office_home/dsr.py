import argparse

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import numpy as np

from dataset import Office, MultiDomainOffice
from office_home.model.modules import DSR

parser = argparse.ArgumentParser(description='VAE office home')
parser.add_argument('--source_domain', type=str, default="Clipart", help='source domain name')
parser.add_argument('--target_domain', type=str, default="Art", help='target domain name')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

source_domain_dataset = Office(type='home', source=args.source_domain, target=args.source_domain)
target_domain_dataset = Office(type='home', source=args.source_domain, target=args.target_domain)

source_loader = DataLoader(source_domain_dataset, batch_size=args.batch_size, shuffle=True)
target_loader = DataLoader(target_domain_dataset, batch_size=args.batch_size, shuffle=True)

model = DSR().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)


def auto_encoder_loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 2048), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + 140 * KLD


def single_classifier_loss_function(pred, label):
    return F.cross_entropy(pred, label)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (source, target) in enumerate(zip(source_loader, target_loader)):
        # for source data
        source_data, source_semantic_label = source

        source_data = source_data.to(device)
        source_semantic_label = source_semantic_label.to(device)
        source_domain_label = torch.zeros_like(source_semantic_label).to(device)

        # for target data
        target_data, target_semantic_label = target

        target_data = target_data.to(device)
        target_semantic_label = target_semantic_label.to(device)
        target_domain_label = torch.zeros_like(target_semantic_label).to(device)

        optimizer.zero_grad()

        # for source auto encoder loss
        source_recon, source_z, source_mu, source_logvar, source_gan = model(source_data)
        source_auto_encoder_loss = auto_encoder_loss_function(source_recon, source_data, source_mu, source_logvar)

        # for target auto encoder loss
        target_recon, target_z, target_mu, target_logvar, target_gan = model(target_data)
        target_auto_encoder_loss = auto_encoder_loss_function(target_recon, target_data, target_mu, target_logvar)

        vae_loss = (source_auto_encoder_loss + target_auto_encoder_loss) / 2

        source_y_d, source_y_y, source_d_d, source_d_y = source_gan
        target_y_d, target_y_y, target_d_d, target_d_y = target_gan

        # y_d classifier loss
        y_d_classifier_result = torch.cat((source_y_d, target_y_d))
        domain_classifier_label = torch.cat((source_domain_label, target_domain_label))
        y_d_classifier_loss = single_classifier_loss_function(y_d_classifier_result, domain_classifier_label)

        # y_y loss only source have y label
        y_y_classifier_loss = single_classifier_loss_function(source_y_y, source_semantic_label)

        # d_d loss
        d_d_classifier_result = torch.cat((source_d_d, target_d_d))
        d_d_classifier_loss = single_classifier_loss_function(d_d_classifier_result, domain_classifier_label)

        # d_y loss, but have no label
        source_d_y_pred = torch.softmax(source_d_y, dim=1)
        target_d_y_pred = torch.softmax(target_d_y, dim=1)

        source_entropy = 5 + torch.mean(torch.sum(torch.log(source_d_y_pred) * source_d_y_pred, dim=1), dim=0)
        target_entropy = 5 + torch.mean(torch.sum(torch.log(target_d_y_pred) * target_d_y_pred, dim=1), dim=0)

        d_y_entropy_loss = (source_entropy + target_entropy) / 2

        total_loss = vae_loss + (y_d_classifier_loss + y_y_classifier_loss + d_d_classifier_loss + d_y_entropy_loss)

        total_loss.backward()

        train_loss += total_loss.item()

        optimizer.step()

        # print('Epoch: {} batch_idx {} loss: {:.4f}'.format(epoch, batch_idx, train_loss))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, label) in enumerate(target_loader):
            data = data.to(device)
            label = label.to(device)
            pred_domain, pred_label = model.predict(data)
            test_loss += single_classifier_loss_function(pred_label, label).item()

            _, predicted_label = torch.max(pred_label, 1)
            total += label.size(0)
            correct += (predicted_label == label).sum().item()

    test_loss /= len(target_loader.dataset)
    print('epoch {} Test set loss: {:.4f}, Accuracy: {:.4f} %%'.format(epoch, test_loss, (100 * correct / total)))
    return 100 * correct / total


if __name__ == "__main__":
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test(epoch)
        best_acc = max(best_acc, acc)
    print(best_acc)

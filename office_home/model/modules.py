import torch
from torch import nn

from torch.nn import functional as F
from torch.autograd import Function


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mu = nn.Linear(2048, 2000)
        self.logvar = nn.Linear(2048, 2000)
        self.mu_batchNorm = nn.BatchNorm1d(2000)
        self.logvar_batchNorm = nn.BatchNorm1d(2000)
        # batch norm

    def forward(self, x):
        mu = self.mu(x)
        mu = self.mu_batchNorm(mu)

        logvar = self.logvar(x)
        logvar = self.logvar_batchNorm(logvar)

        return mu, logvar


class DualDecoder(nn.Module):
    def __init__(self):
        super(DualDecoder, self).__init__()
        self.fc = nn.Linear(4000, 2048)
        self.input_layer_norm = nn.LayerNorm(normalized_shape=4000)
        self.out_layer_norm = nn.LayerNorm(normalized_shape=2048)
        self.drop = nn.Dropout()

    def forward(self, z):
        z = self.input_layer_norm(z)
        h = F.relu(self.fc(z))
        h = self.drop(h)
        h = self.out_layer_norm(h)
        return h


class GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class LabelGAN(nn.Module):
    def __init__(self):
        super(LabelGAN, self).__init__()
        self.grl = GradientReversal(lambda_=0.15)

        self.domain_classifier = nn.Sequential(
            nn.Linear(2000, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.9),
            nn.BatchNorm1d(400),
            nn.Linear(400, 2),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(2000, 3000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(3000),
            nn.Linear(3000, 65)
        )

    def forward(self, z):
        grl = self.grl(z)
        return self.domain_classifier(grl), self.label_classifier(z)


class DomainGAN(nn.Module):
    def __init__(self):
        super(DomainGAN, self).__init__()
        self.grl = GradientReversal(lambda_=0.15)

        self.domain_classifier = nn.Sequential(
            nn.Linear(2000, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 2),
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 65)
        )

    def forward(self, z):
        grl = self.grl(z)
        return self.domain_classifier(z), self.label_classifier(grl)


class DSR(nn.Module):
    def __init__(self):
        super(DSR, self).__init__()
        self.domain_encoder = Encoder()
        self.label_encoder = Encoder()

        self.decoder = DualDecoder()

        self.domain_gan = DomainGAN()
        self.label_gan = LabelGAN()
        # decoder

    def forward(self, x):
        x = x.view(-1, 2048)

        #### encoder
        mu_y, logvar_y = self.label_encoder(x)
        mu_d, logvar_d = self.domain_encoder(x)

        mu = torch.cat((mu_y, mu_d), dim=1)
        logvar = torch.cat((logvar_y, logvar_d), dim=1)

        #### get z
        z_y = self.reparameterize(mu_y, logvar_y)
        z_d = self.reparameterize(mu_d, logvar_d)

        z = torch.cat((z_y, z_d), dim=1)

        z_y_d, z_y_y = self.label_gan(z_y)
        z_d_d, z_d_y = self.domain_gan(z_d)

        #### decoder
        recon_x = self.decoder(z)
        return recon_x, z, mu, logvar, (z_y_d, z_y_y, z_d_d, z_d_y)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict(self, x):
        # 预测的时候，是直接encode x之后，然后送到label classifier里，然后分类
        x = x.view(-1, 2048)

        # encoder
        mu_y, logvar_y = self.label_encoder(x)
        z_y = self.reparameterize(mu_y, logvar_y)
        z_y_d, z_y_y = self.label_gan(z_y)
        return z_y_d, z_y_y

import numpy as np
import torch
from torch import optim
from torch.nn import functional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ctgan.conditional import ConditionalGenerator
from ctgan.models import Discriminator, Generator
from ctgan.sampler import Sampler
from ctgan.transformer import DataTransformer


class CTGANSynthesizer(object):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        gen_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Resiudal Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        dis_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        l2scale (float):
            Wheight Decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
    """

    def __init__(self, demand_col, embedding_dim=128, gen_dim=(256, 256), dis_dim=(256, 256),
                 l2scale=1e-6, batch_size=500):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.demand_column = demand_col

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def plot_grad_flow(self, named_parameters, axis, plot_title):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        axis.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        axis.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        axis.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        axis.set_xticklabels(layers, rotation="vertical")
        axis.set_xlim(left=0, right=len(ave_grads))
        axis.set_ylim(bottom=-0.001, top=0.2)  # zoom in on the lower gradient regions
        axis.set_xlabel("Layers")
        axis.set_ylabel("average gradient")
        axis.set_title(plot_title)
        plt.grid(True)
        axis.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)],
                   ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def fit(self, train_data, eval_interval, discrete_columns=tuple(), epochs=300, log_frequency=True):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a
                pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of training epochs. Defaults to 300.
            log_frequency (boolean):
                Whether to use log frequency of categorical levels in conditional
                sampling. Defaults to ``True``.
        """

        train = train_data
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data,
            self.transformer.output_info,
            log_frequency
        )

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim
        ).to(self.device)

        self.discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim
        ).to(self.device)

        optimizerG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9),
            weight_decay=self.l2scale
        )
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        stats_real = train[self.demand_column].describe()
        stats_real_week = train.groupby('Weekday')[self.demand_column].describe()
        stats_real_month = train.groupby('Month')[self.demand_column].describe()

        # figure for plotting gradients; ax1 for generator and ax2 for discriminator
        fig, (ax1, ax2) = plt.subplots(1, 2)

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)

                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = self.discriminator(fake_cat)
                y_real = self.discriminator(real_cat)

                pen = self.discriminator.calc_gradient_penalty(real_cat, fake_cat, self.device)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()

                #plot gradients
                self.plot_grad_flow(self.discriminator.named_parameters(), ax2, 'Gradient flow for D')

                optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()

                # plot gradients
                self.plot_grad_flow(self.generator.named_parameters(), ax1, 'Gradient flow for G')

                optimizerG.step()


            print("Epoch %d, Loss G: %.4f, Loss D: %.4f" %
                  (i + 1, loss_g.detach().cpu(), loss_d.detach().cpu()),
                  flush=True)

            #check model results every x epochs

            if (i+1)%eval_interval == 0:
                eval_sample = self.sample(1000)
                sample = pd.DataFrame(eval_sample, columns=eval_sample.columns)
                sample.loc[:, self.demand_column].hist(bins=50, alpha=0.4, label= 'fake')
                pd.DataFrame(train, columns=train.columns).loc[:, self.demand_column].hist(bins=50, alpha=0.4, label='real')
                plt.legend()
                plt.show()

                print((sample[self.demand_column].describe()-stats_real)/stats_real)
                print(' ')
                print(((sample.groupby('Weekday')[self.demand_column].describe() - stats_real_week)/stats_real_week).T)
                print(' ')
                print(((sample.groupby('Month')[self.demand_column].describe() - stats_real_month) / stats_real_month).T)

        plt.show()




    def sample(self, n):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        self.generator.eval()
        steps = n // self.batch_size + 1
        data = []
        np.random.seed(0)
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        self.generator.train()

        return self.transformer.inverse_transform(data, None)

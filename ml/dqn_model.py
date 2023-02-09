import torch
import torch.nn as nn

import numpy as np


class QNet(nn.Module):
    def __init__(self, input_shape, n_actions, state_size):
        super(QNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # print("{} + {} => {}".format(conv_out_size, state_size, 600))
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + state_size, 600),
            nn.ReLU(),
            nn.Linear(600, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, state):
        conv_out = self.conv(x).view(x.size()[0], -1)
        # print('conv: {}, state: {}'.format(conv_out.dtype, state.dtype))
        combined_state = torch.cat((conv_out, state), axis=1)
        # print('Combined_state shape: {}'.format(combined_state.shape))
        return self.fc(combined_state)

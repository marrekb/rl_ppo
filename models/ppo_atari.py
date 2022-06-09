import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init import init_orthogonal_head, init_orthogonal_features

class PolicyValueModel(nn.Module):
    def __init__(self, count_of_actions, count_of_features, 
                 value_support_size = 1):
        super(PolicyValueModel, self).__init__()

        self.features_model = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3,
                      stride = 1, padding = 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(count_of_features, 512),
            nn.ReLU()
        )
        self.features_model.apply(init_orthogonal_features)

        self.policy_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, count_of_actions)
        )
        self.policy_model.apply(init_orthogonal_head)

        self.value_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, value_support_size if value_support_size == 1 else value_support_size * 2 + 1)
        )
        self.value_model.apply(init_orthogonal_head)

    def forward(self, x):
        x = self.features_model(x)
        return self.policy_model(x), self.value_model(x)

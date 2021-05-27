import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesformer.models.vit import TimeSformer


class ITR_v1(nn.Module):
    def __init__(self, in_features=768*2, nb_classes=2, nb_layers=3):
        super(ITR_v1, self).__init__()
        self.params_file = "/home/acances/Code/human_interaction_ITR/params/TimeSformer_divST_8x32_224_K400.pyth"
         
        self.timesformer = TimeSformer(
            img_size=224, num_classes=400, num_frames=8,
            attention_type='divided_space_time',
            pretrained_model=self.params_file
        )

        self.in_features = in_features
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        feature_sizes = [self.in_features // 2**i for i in range(self.nb_layers)] + [self.nb_classes]
        self.layers = nn.ModuleList([nn.Linear(feature_sizes[i], feature_sizes[i + 1]) for i in range(self.nb_layers)])
    
    def forward(self, input1, input2):
        features1 = self.timesformer.model.forward_features(input1)
        features2 = self.timesformer.model.forward_features(input2)

        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        x = torch.cat((features1, features2), dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x


if __name__ == "__main__":
    model = ITR_v1()
    model.cuda()

    torch.manual_seed(0)
    input1 = torch.rand(1, 3, 8, 224, 224).cuda()
    input2 = torch.rand(1, 3, 8, 224, 224).cuda()

    output = model(input1, input2)
    print(input1.shape)
    print(output.shape)


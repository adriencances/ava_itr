import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesformer.models.vit import TimeSformer


class ITR(nn.Module):
    def __init__(self):
        super(ITR, self).__init__()
        self.params_file = "/home/acances/Code/human_interaction_ITR/params/TimeSformer_divST_8x32_224_K400.pyth"
         
        self.timesformer = TimeSformer(
            img_size=224, num_classes=400, num_frames=8,
            attention_type='divided_space_time',
            pretrained_model=self.params_file
        )
    
    def forward(self, input1, input2):
        output1 = self.timesformer.model.forward_features(input1)
        output2 = self.timesformer.model.forward_features(input2)
        
        return output1, output2


if __name__ == "__main__":
    model = ITR()
    model.cuda()
    # print(model.size)

    torch.manual_seed(0)
    input1 = torch.rand(1, 3, 8, 224, 224).cuda()
    input2 = torch.rand(1, 3, 8, 224, 224).cuda()

    output1, output2 = model(input1, input2)
    print(output1.shape)
    print(output2.shape)

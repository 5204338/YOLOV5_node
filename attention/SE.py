import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self,channel,ratio = 16):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//ratio,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//ratio,channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,h,w = x.size()
        #平均池化 b,c,w,h -> b,c,1,1
        avg = self.avg_pool(x).view([b,c])

        #平均池化 b,c-> b,c,1,1
        fc = self.fc(avg).view([b,c,1,1])
        #print(fc)
        return x * fc

model = se_block(512)
print(model)
input = torch.ones([2,512,26,26])
outputs = model(input)
print(outputs)


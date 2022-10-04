class MaxOut2D(nn.Module):
    
    def __init__(self, max_out, dim=None):
        super(MaxOut2D, self).__init__()
        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)
        # self.weight = nn.Parameter(torch.Tensor(torch.ones([1,1,dim])))
        # self.bias = nn.Parameter(torch.Tensor(torch.ones([1,1,dim])))

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]

        # Reshape input from N x C x H x W --> N x H*W x C
        x_reshape = torch.permute(x, (0, 2, 3, 1)).view(batch_size, height * width, channels)
        
        # weight & bias
        # x_reshape = x_reshape*F.gelu(self.weight.repeat(batch_size, height * width,1))+F.gelu(self.bias.repeat(batch_size, height * width,1))
       
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        
        # Reshape back to N x C//maxout_kernel x H x W.
        return torch.permute(x_pooled, (0, 2, 1)).view(batch_size, channels // self.max_out, height, width).contiguous()

class AvgOut2D(nn.Module):
    
    def __init__(self, max_out, dim=None):
        super(AvgOut2D, self).__init__()
        self.max_out = max_out
        self.avg_pool = nn.AvgPool1d(max_out)
        # self.weight = nn.Parameter(torch.Tensor(torch.ones([1,1,dim])))
        # self.bias = nn.Parameter(torch.Tensor(torch.ones([1,1,dim])))

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        
        # Reshape input from N x C x H x W --> N x H*W x C
        x_reshape = torch.permute(x, (0, 2, 3, 1)).view(batch_size, height * width, channels)
        
        # weight & bias
        # x_reshape = x_reshape*F.gelu(self.weight.repeat(batch_size, height * width,1))+F.gelu(self.bias.repeat(batch_size, height * width,1))
        
        # Pool along channel dims
        x_pooled = self.avg_pool(x_reshape)
        
        # Reshape back to N x C//maxout_kernel x H x W.
        return torch.permute(x_pooled, (0, 2, 1)).view(batch_size, channels // self.max_out, height, width).contiguous()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.maxout = MaxOut2D(2,self.expansion*planes)
        self.avgout = AvgOut2D(2,self.expansion*planes)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.fc2 = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))

    def forward(self, x, out):
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        ref = x.cuda()

        input1 = self.avg_pool(out)
        input2 = self.avg_pool(ref)

        tmp = torch.stack((input1,input2), dim=2)
        tmp = torch.flatten(tmp, start_dim=1, end_dim=2)

        max_out = self.fc(self.maxout(tmp))

        avg_out = self.fc2(self.avgout(tmp))

        tmp = max_out + avg_out
        tmp = torch.sigmoid(tmp)
        out = out * (tmp)
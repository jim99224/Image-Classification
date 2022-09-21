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
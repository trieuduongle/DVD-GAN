import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

from Module.Normalization import ConditionalNorm, SpectralNorm 
from Module.Attention import SelfAttention
from Module.GResBlock import GResBlock



class Spectral_Norm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = Spectral_Norm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    Spectral_Norm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)


def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


def leaky_relu(input):
    return F.leaky_relu(input, negative_slope=0.2)



class SpatialDiscriminator(nn.Module):
    def __init__(self, chn=128, n_class=3):
        super().__init__()

        self.chn = chn

        def conv(in_channel, out_channel, downsample=True):
            return GResBlock(in_channel, out_channel,
                          bn=False,
                          upsample_factor=False, downsample_factor=downsample)

        gain = 2 ** 0.5
        

        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv2d(3, 2*chn, 3,padding=1),),
                                      nn.ReLU(),
                                      SpectralNorm(nn.Conv2d(2*chn, 2*chn, 3,padding=1),),
                                      nn.AvgPool2d(2))
        self.pre_skip = SpectralNorm(nn.Conv2d(3, 2*chn, 1))

        self.conv1 = conv(2*chn, 4*chn, downsample=True)
        self.attn = SelfAttention(4*chn)
        self.conv2 = nn.Sequential(
                        conv(4*chn, 8*chn, downsample=True),    
                        conv(8*chn, 16*chn, downsample=True),
                        conv(16*chn, 16*chn, downsample=True))

        self.linear = SpectralNorm(nn.Linear(16*chn, 1))

        self.embed = nn.Embedding(n_class, 16*chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, x, class_id):
        
        # reshape input tensor from BxTxCxHxW to BTxCxHxW
        batch_size, T, C, W, H = x.size()
        print(x.size())

        x = x.view(batch_size*T, C, H, W)

        out = self.pre_conv(x)
        out = out + self.pre_skip(F.avg_pool2d(x, 2))

        # reshape back to BxTxCxHxW

        out = out.view(batch_size, T, -1, H // 2, W // 2)

        out = self.conv1(out) # BxTxCxHxW
        out = out.permute(0, 2, 1, 3, 4) # BxCxTxHxW

        out = self.attn(out) # BxCxTxHxW
        out = out.permute(0, 2, 1, 3, 4).contiguous() # BxTxCxHxW

        out = self.conv2(out)

        print("after conv:", out.size())

        out = F.relu(out)
        out = out.view(batch_size, T, out.size(2), -1)

        # sum on H and W axis
        out = out.sum(3)
        # sum on T axis
        out = out.sum(1)

        out_linear = self.linear(out).squeeze(1)

        embed = self.embed(class_id)

        prod = (out * embed).sum(1)


        return out_linear + prod

def sample_k_frames(data, length, n_frame):
    idx = torch.randint(0, length, n_frame)
    # idx = torch.LongTensor(random.sample(range(0, length), n_frame)).cuda()
    idx = idx.sort()

    return data[:, idx[0], :, :, :]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class Res3dBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=False):
        super(Res3dBlock, self).__init__()
        self.conv1 = conv3x3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample
        if self.downsample:
            self.conv_sc = nn.Sequential(
                    nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False), 
                    nn.BatchNorm3d(out_channel))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.conv_sc(x)

        out += residual
        out = self.relu(out)

        return out


class TemporalDiscriminator(nn.Module):
    def __init__(self, chn=128, n_class=3):
        super().__init__()

        def conv(in_channel, out_channel, downsample=True):
            return GResBlock(in_channel, out_channel,
                          bn=False,
                          upsample_factor=False, downsample_factor=downsample)

        gain = 2 ** 0.5
        
        self.pre_conv = nn.Sequential(SpectralNorm(nn.Conv3d(3, 2*chn, 3,padding=1),),
                                      nn.ReLU(),
                                      SpectralNorm(nn.Conv3d(2*chn, 2*chn, 3,padding=1),),
                                      nn.AvgPool3d(2))
        self.pre_skip = SpectralNorm(nn.Conv3d(3, 2*chn, 1))


        self.res3d = Res3dBlock(2*chn, 4*chn, downsample=True)

        self.conv = nn.Sequential(SelfAttention(4*chn),
                                  conv(4*chn, 8*chn, downsample=True),    
                                  conv(8*chn, 16*chn, downsample=True),
                                  conv(16*chn, 16*chn, downsample=True))

        self.linear = SpectralNorm(nn.Linear(16*chn, 1))

        self.embed = nn.Embedding(n_class, 16*chn)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.embed = spectral_norm(self.embed)

    def forward(self, x, class_id):

        # pre-process with avg_pool2d to reduce tensor size
        B, T, C, H, W = x.size()
        x = F.avg_pool2d(x.view(B*T, C, H, W), kernel_size=2)
        _, _, H, W = x.size()
        x = x.view(B, T, C, H, W)

        #transpose to BxCxTxHxW
        x = x.permute(0, 2, 1, 3, 4)

        out = self.pre_conv(x)
        # print(out.size())
        # print(out.type())
        out = out + self.pre_skip(F.avg_pool3d(x, 2))
        # print(out.size())
        # print(out.type())
        out = self.res3d(out) # BxCxTxHxW
      
        # out = out.permute(0, 2, 1, 3, 4)
        # print(out.size())
        # print(out.type())

        # B, T, C, H, W = out.size()
        # out = out.view(B*T, C, H, W)
        # exit()



        # print(out.size())
        out = self.conv(out)
        # print(out.size())
        # print(out.type())
        out = F.relu(out)
        # print(out.size())
        # print(out.type())
        # out = out.view(out.size(0), out.size(1), -1)
        out = out.view(B, T, out.size(1), -1)
        # print(out.size())
        # print(out.type())
        # sum on H and W axis
        out = out.sum(3)
        # print(out.size())
        # sum on T axis
        out = out.sum(1)
        # print(out.size())
        # print(out.type())
        out_linear = self.linear(out).squeeze(1)
        # print(out_linear.size())
        # print(out_linear.type())

        
        embed = self.embed(class_id)
        # print(embed.size())
        # print(embed.type())

        prod = (out * embed).sum(1)

        return out_linear + prod

def main():
    # m = torch.randn((3, 5, 1, 1, 1))
    # n = torch.LongTensor(random.sample(range(0, 4), 2))
    # l = m[:,n,:,:,:]


    # m = torch.randn(5)
    # n = m.view(-1, 1)
    # l = n.repeat(1, 2) # 2 number of repeat
    # l = l.view(-1)
    model = TemporalDiscriminator()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0, 0.9),
                                               weight_decay=0.00001)
    for i in range(100):
        data = torch.randn((3, 5, 3, 64, 64)).cuda()

        label = torch.randint(0, 3, 3)
        # label = torch.cuda.LongTensor(random.sample(range(0,3), 3))

        # data = sample_k_frames(data, data.size(1), 3)

        B, T, C, H, W = data.size()
        data = F.avg_pool2d(data.view(B*T, C, H, W), kernel_size=2)
        _, _, H, W = data.size()
        # print(data.size())
        # print(data.type())
        data = data.view(B, T, C, H, W)

        #transpose to BxCxTxHxW
        data = data.transpose(1, 2).contiguous()


        out = model(data, label)

        # print(out.type())
        # print(out.size())

        loss = torch.mean(out)

        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
  
  main()


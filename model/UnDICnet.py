import torch
import torch.nn as nn
import torch.nn.functional as F
import collections

class Corr_pyTorch(nn.Module):
    '''
    my implementation of correlation layer using pytorch
    note that the Ispeed is much slower than cuda version
    '''

    def __init__(self, pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.stride1 = stride1
        self.stride2 = stride2
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(pad_size, 0)


    
    def forward(self, in1, in2):
        bz, cn, hei, wid = in1.shape
        f1 = F.unfold(in1, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride1)
        f2 = F.unfold(in2, kernel_size=self.kernel_size, padding=self.kernel_size // 2,stride=self.stride2)
        searching_kernel_size = f2.shape[1]
        f2_ = torch.reshape(f2, (bz, searching_kernel_size, hei, wid))
        f2_ = torch.reshape(f2_, (bz * searching_kernel_size, hei, wid)).unsqueeze(1)
        f2 = F.unfold(f2_, kernel_size=(hei, wid), padding=self.pad_size, stride=self.stride2)
        _, kernel_number, window_number = f2.shape
        f2_ = torch.reshape(f2, (bz, searching_kernel_size, kernel_number, window_number))
        f2_2 = torch.transpose(f2_, dim0=1, dim1=3).transpose(2, 3)
        f1_2 = f1.unsqueeze(1)

        res = f2_2 * f1_2
        res = torch.mean(res, dim=2)
        res = torch.reshape(res, (bz, window_number, hei, wid))
        return res

class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()
    def forward(self, x, deformation):
        B, C, H, W = x.shape
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)

        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float().to(device=x.device)
        vgrid = grid + deformation  # B,2,H,W

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2，
        x_warp = F.grid_sample(x, vgrid, padding_mode='zeros', align_corners=True)
        mask = torch.ones(x.size(), requires_grad=False).to(x.device)
        mask = F.grid_sample(mask, vgrid, align_corners=True)
        mask = (mask >= 1.0).float()
        return x_warp * mask



def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, if_IN=False, IN_affine=False, if_BN=False):
    if isReLU:
        if if_IN:

            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)

            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True)
            )
    else:
        if if_IN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.InstanceNorm2d(out_planes, affine=IN_affine)
            )
        elif if_BN:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.BatchNorm2d(out_planes, affine=IN_affine)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                          padding=((kernel_size - 1) * dilation) // 2, bias=True)
            )


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(inplace=False, negative_slope=0.1)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    
    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)
class Encoder(nn.Module):
    def __init__(self, input_dim=1 ,norm_fn='batch', dropout=0.0):
        super(Encoder, self).__init__()
        self.norm_fn = norm_fn
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=16)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(16)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(16)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=7, stride=2, padding=3)
        self.relu1 =nn.LeakyReLU(inplace=False, negative_slope=0.1)
        self.in_planes =16

        self.layer1 = self._make_layer(16, stride=1)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(64, stride=2)
        self.layer4 = self._make_layer(96, stride=2)
        self.layer5 = self._make_layer(128, stride=2)
        self.layer6= self._make_layer(196, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        disp1 = self.layer1(x)
        disp2 = self.layer2(disp1)
        disp3 = self.layer3(disp2)
        disp4 = self.layer4(disp3)
        disp5= self.layer5(disp4)
        disp6 = self.layer6(disp5)
        return disp6, disp5,disp4, disp3, disp2,


class Denseblock(nn.Module):

    def __init__(self, ch_in, f_channels=(128, 128, 96, 64, 32), out_channel=2):
        super(Denseblock, self).__init__()

        N = 0
        ind = 0
        N += ch_in
        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]

        ind += 1
        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.n_channels = N
        ind += 1
        self.conv_last = conv(N, out_channel, isReLU=False)
        self.channels = (128, 128, 128, 96, 64, 32, 2)
        self.convs = nn.Sequential(
            conv(self.n_channels+2, self.channels[0], 3, 1, 1),
            conv(self.channels[0], self.channels[1], 3, 1, 2),
            conv(self.channels[1], self.channels[2], 3, 1, 4),
            conv(self.channels[2], self.channels[3], 3, 1, 8),
            conv(self.channels[3], self.channels[4], 3, 1, 16),
            conv(self.channels[4], self.channels[5], 3, 1, 1),
            conv(self.channels[5], self.channels[6], isReLU=False)
        )
    def forward(self,disp, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x4 = torch.cat([self.conv4(x3), x3], dim=1)
        x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x5)
        disp_ = disp + x_out
        out=self.convs(torch.cat([x5, disp_], dim=1))
        out_=out + x_out
        return disp,out_


class DICnet(nn.Module):
    def __init__(self):
        super(DICnet, self).__init__()
        self.num_chs = [1, 16, 32, 64, 96, 128, 196]  # 1/2 1/4 1/8 1/16 1/32 1/64
        self.feature_pyramid_extractor = Encoder()
        self.conv_1x1 = nn.ModuleList([conv(196, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(128, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(96, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(64, 32, kernel_size=1, stride=1, dilation=1),
                                       conv(32, 32, kernel_size=1, stride=1, dilation=1)])

        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)
        self.warping_layer = WarpingLayer()
        self.search_range = 4
        self.output_level = 4
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = self.dim_corr + 32 + 2
        self.d_channels = (128, 128, 96, 64, 32)
        self.correlation_pytorch = Corr_pyTorch(pad_size=self.search_range, kernel_size=1,max_displacement=self.search_range, stride1=1,stride2=1)  # correlation layer using pytorch
        self.denseblock = Denseblock(self.num_ch_in, f_channels=self.d_channels)

    def upsample2d_as(self,inputs, target_as, mode="bilinear", if_rate=False):
        _, _, h, w = target_as.size()
        res = F.interpolate(inputs, [h, w], mode=mode, align_corners=True)
        if if_rate:
            _, _, h_, w_ = inputs.size()
            u_scale = (w / w_)
            v_scale = (h / h_)
            u, v = res.chunk(2, dim=1)
            u = u * u_scale
            v = v * v_scale
            res = torch.cat([u, v], dim=1)
        return res
    def normalize_features(self, feature_list, normalize, center, moments_across_channels=False,
                           moments_across_images=False):
        """Normalizes feature tensors (e.g., before computing the cost volume).
        Args:
          feature_list: list of torch tensors, each with dimensions [b, c, h, w]
          normalize: bool flag, divide features by their standard deviation
          center: bool flag, subtract feature mean
          moments_across_channels: bool flag, compute mean and std across channels, 看到UFlow默认是True
          moments_across_images: bool flag, compute mean and std across images, 看到UFlow默认是True

        Returns:
          list, normalized feature_list
        """
        statistics = collections.defaultdict(list)
        axes = [1, 2, 3] if moments_across_channels else [2, 3]  # [b, c, h, w]
        for feature_image in feature_list:
            mean = torch.mean(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            variance = torch.var(feature_image, dim=axes, keepdim=True)  # [b,1,1,1] or [b,c,1,1]
            statistics['mean'].append(mean)
            statistics['var'].append(variance)
        if moments_across_images:
            statistics['mean'] = ([torch.mean(torch.stack(statistics['mean'], dim=0), dim=(0,))] * len(feature_list))
            statistics['var'] = ([torch.var(torch.stack(statistics['var'], dim=0), dim=(0,))] * len(feature_list))
        statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]
        if center:
            feature_list = [
                f - mean for f, mean in zip(feature_list, statistics['mean'])
            ]
        if normalize:
            feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

        return feature_list
    def Update(self, level, flow_1,  feature_1, feature_1_1x1, feature_2):
        up_bilinear = self.upsample2d_as(flow_1, feature_1, mode="bilinear", if_rate=True)
        if level == 0:
            feature_2_warp = feature_2
        else:
            feature_2_warp = self.warping_layer(feature_2, up_bilinear)
        feature_1, feature_2_warp = self.normalize_features((feature_1, feature_2_warp), normalize=True,center=True)
        out_corr_1 = self.correlation_pytorch(feature_1, feature_2_warp)
        out_corr_relu_1 = self.leakyRELU(out_corr_1)
        up_bilinear, res = self.denseblock(up_bilinear ,torch.cat([out_corr_relu_1, feature_1_1x1, up_bilinear], dim=1))
        return  up_bilinear+res

    def forward(self,img1,img2):
        x1 = self.feature_pyramid_extractor(img1)
        x2 = self.feature_pyramid_extractor(img2)
        b_size, _, h_x1, w_x1, = x1[0].size()
        init_dtype = x1[0].dtype
        init_device = x1[0].device
        disp_f= torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        feature_level_ls = []
        for l, (x1,x2) in enumerate(zip(x1,x2)):
            x1_1by1 = self.conv_1x1[l](x1)
            feature_level_ls.append((x1, x1_1by1, x2))  # len = 5
        for level, (x1, x1_1by1, x2) in enumerate(feature_level_ls):
            disp_f = self.Update(level=level, flow_1=disp_f,feature_1=x1, feature_1_1x1=x1_1by1,feature_2=x2)
        disp = self.upsample2d_as(disp_f, img1, mode="bilinear", if_rate=True)
        return disp

class UnDICnet_d(nn.Module):
    def __init__(self,args):
        super(UnDICnet_d, self).__init__()
        self.DICnet = DICnet()
        self.args = args
    def forward(self, img1,img2):

        d_f_out = self.DICnet(img1,img2)  # forward estimation
        d_b_out = self.DICnet(img2,img1)  # backward estimation
        output_dict = {}

        output_dict['flow_f_out'] = d_f_out
        output_dict['flow_b_out'] = d_b_out
        return output_dict


class UnDICnet_s(nn.Module):
    def __init__(self, args):
        super(UnDICnet_s, self).__init__()
        self.DICnet = DICnet()
        self.args = args

    def forward(self, img1, img2):
        d_f_out = self.DICnet(img1, img2)  # forward estimation
        return d_f_out

if __name__ == '__main__':
    model =UnDICnet_d(args=None)
    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    img1 = torch.randn(1, 1, 256, 256)
    img2 = torch.randn(1, 1, 256, 256)
    model(img1, img2)


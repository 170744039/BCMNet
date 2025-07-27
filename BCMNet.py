import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x


def get_topk(x, k=10, dim=-3):
    # b, c, h, w = x.shape
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class mobile_Self_Correlation_Per(nn.Module):
    # input:[?,512,32,32] out:[?,115,32,32]
    def __init__(self, nb_pools=15,patch_size=1):
        super(mobile_Self_Correlation_Per, self).__init__()
        self.nb_pools = nb_pools
        self.patch_size = patch_size

    def forward(self, x):
        patch_size = self.patch_size
        b,c,h,w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
        num_patch_h = int(h/patch_size)
        num_patch_w = int(w/patch_size)
        num_patches = num_patch_w*num_patch_h
        patch_area = patch_size*patch_size   # 一个patch包含几个像素点
        assert num_patches == (h*w)/patch_area


        x = F.normalize(x, p=2, dim=-3)  # 在通道维度归一化


        # avg = x.mean(axis=-3, keepdim=True)
        # std = x.std(axis=-3, keepdim=True)
        # x = (x - avg) / std

        # 下面的N指patch个数num_patches，P指一个patch包含几个像素点patch_area
        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(b * c * num_patch_h, patch_size, num_patch_w, patch_size)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(b, c, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(b * patch_area, num_patches, -1)

        x = torch.matmul(x, x.transpose(1, 2))  # 相似度矩阵[BP, N, N]

        # x = x.contiguous().view(b, num_patches, num_patch_h, num_patch_w)

        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(b, patch_area, num_patches, -1)
        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(b * num_patches * num_patch_h, num_patch_w, patch_size, patch_size)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(b, num_patches, num_patch_h * patch_size, num_patch_w * patch_size)

        x = get_topk(x, k=self.nb_pools, dim=-3)  # 在h1*w1维度选取top_k

        return x  # (b, top_k, h, w)


class BCMNet(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(BCMNet, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        self.pool_1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)


        self.corr1 = mobile_Self_Correlation_Per(nb_pools=512)
        # self.corr1 = Self_Correlation_Per(nb_pools=512)


        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=512, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.CONV0_3 = ContinusParalleConv(64, 64, pre_Batch_Norm=True)

        # 分割头

        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )


    def forward(self, x):
        x_0_0 = self.stage_0(x)

        x_1_0 = self.pool_1(x_0_0)

        x_1_0 = self.stage_1(x_1_0)

        x_1_0_corr = self.corr1(x_1_0)

        x_0_3 = self.upsample_0_3(x_1_0_corr)
        x_0_3 = self.CONV0_3(x_0_3)

        if self.deep_supervision:
            # out_put1 = self.final_super_0_1(x_2_1)
            # out_put2 = self.final_super_0_2(x_1_2)
            out_put3 = self.final_super_0_3(x_0_3)
            return out_put3
        else:
            return self.final_super_0_3(x_0_3)

    def forward_corr(self, x):

        x_0_0 = self.stage_0(x)
        # print(x_0_0.shape,111)

        x_1_0 = self.pool_1(x_0_0)
        # print(x_1_0.shape)

        x_1_0 = self.stage_1(x_1_0)

        x_1_0_corr = self.corr1(x_1_0)

        return x_1_0_corr





if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = torchvision.transforms.Compose([transforms.ToPILImage(), transforms.Resize((320, 320)),
                                                torchvision.transforms.ToTensor(),
                                                # transforms.ColorJitter(brightness=0.3, contrast=0.3),
                                                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                #                          std=[0.229, 0.224, 0.225]),
                                                ])



    image = cv2.imread(r"C:\Users\tangguo\Desktop\comofod\128_F.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transform(image)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).cuda()

    deep_supervision = True
    model = BCMNet(num_classes=1, deep_supervision=deep_supervision)

    # model.load_state_dict(torch.load(r'D:\weights\cisa_test\model_17_148000_0.13829921185970306.pth'))
    model = model.to(device)
    pred = model(img)
    print(pred.shape, 111)


    # model_features = torch.nn.Sequential(*(list(model.children())[:6]))
    #
    #
    # print(model_features)
    # model_features.eval()
    # model.eval()
    # with torch.no_grad():  # 不计算梯度
    #     feature_map = model_features(img)

    # feature_map_avg = torch.mean(feature_map, dim=1)
    #
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(feature_map_avg[0].cpu().numpy(), cmap='gray')  # 可视化第一个过滤器的输出
    # plt.colorbar()
    # plt.show()






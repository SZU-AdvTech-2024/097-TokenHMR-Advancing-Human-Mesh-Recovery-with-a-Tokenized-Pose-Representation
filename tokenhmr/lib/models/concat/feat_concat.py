import torch
import torch.nn as nn

class AdditiveFusion(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        super(AdditiveFusion, self).__init__()
        # 定义一个全连接层，将 depth_feats 从 feat_dim 映射到 embed_dim
        self.fc = nn.Linear(feat_dim, embed_dim)

    def forward(self, conditioning_feats, depth_feats):
        # depth_feats 是一个 (B, feat_dim) 的向量
        # 通过全连接层将其投影到 embed_dim 维度，得到 (B, embed_dim)
        depth_feats_proj = self.fc(depth_feats)

        # 将投影后的 depth_feats_proj 变成 (B, embed_dim, 1, 1) 的形状
        # 以便与 conditioning_feats 进行广播相加
        depth_feats_proj = depth_feats_proj.unsqueeze(-1).unsqueeze(-1)

        # 将 depth_feats_proj 扩展成与 conditioning_feats 相同的空间维度 (Hp, Wp)
        depth_feats_proj = depth_feats_proj.expand(-1, -1, conditioning_feats.shape[2], conditioning_feats.shape[3])

        # 将 depth_feats_proj 和 conditioning_feats 相加，得到融合后的特征
        return conditioning_feats + depth_feats_proj

class ConcatConvFusion(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        super(ConcatConvFusion, self).__init__()
        # 定义一个全连接层，将 depth_feats 从 feat_dim 映射到 embed_dim
        self.fc = nn.Linear(feat_dim, embed_dim)

        # 定义一个 1x1 卷积层，将拼接后的特征 (2*embed_dim) 映射回 embed_dim
        self.conv = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(self, conditioning_feats, depth_feats):
        # 通过全连接层将 depth_feats 投影到 embed_dim 维度，得到 (B, embed_dim)
        depth_feats_proj = self.fc(depth_feats)

        # 将 depth_feats_proj 变成 (B, embed_dim, 1, 1) 的形状
        # 以便与 conditioning_feats 在通道维度上进行拼接
        depth_feats_proj = depth_feats_proj.unsqueeze(-1).unsqueeze(-1)

        # 扩展 depth_feats_proj 为 (B, embed_dim, Hp, Wp) 以匹配 conditioning_feats
        depth_feats_proj = depth_feats_proj.expand(-1, -1, conditioning_feats.shape[2], conditioning_feats.shape[3])

        # 在通道维度上拼接两个特征，得到 (B, 2*embed_dim, Hp, Wp)
        concatenated_feats = torch.cat([conditioning_feats, depth_feats_proj], dim=1)

        # 通过 1x1 卷积将拼接后的特征从 2*embed_dim 压缩回 embed_dim
        return self.conv(concatenated_feats)

class MultiplicativeFusion(nn.Module):
    def __init__(self, feat_dim, embed_dim):
        super(MultiplicativeFusion, self).__init__()
        # 定义一个全连接层，将 depth_feats 从 feat_dim 映射到 embed_dim
        self.fc = nn.Linear(feat_dim, embed_dim)

    def forward(self, conditioning_feats, depth_feats):
        # 通过全连接层将 depth_feats 投影到 embed_dim 维度，得到 (B, embed_dim)
        depth_feats_proj = self.fc(depth_feats)

        # 将 depth_feats_proj 变成 (B, embed_dim, 1, 1) 的形状
        # 以便与 conditioning_feats 逐元素相乘
        depth_feats_proj = depth_feats_proj.unsqueeze(-1).unsqueeze(-1)

        # 将 depth_feats_proj 扩展为与 conditioning_feats 相同的空间维度 (Hp, Wp)
        depth_feats_proj = depth_feats_proj.expand(-1, -1, conditioning_feats.shape[2], conditioning_feats.shape[3])

        # 将 depth_feats_proj 和 conditioning_feats 逐元素相乘，得到融合后的特征
        return conditioning_feats * depth_feats_proj

def get_concat(fusion_type, feat_dim, embed_dim):
    # 融合方式字典，将字符串标签映射到相应的融合类
    fusion_dict = {
        'additive': AdditiveFusion(feat_dim, embed_dim),
        'concat_conv': ConcatConvFusion(feat_dim, embed_dim),
        'multiplicative': MultiplicativeFusion(feat_dim, embed_dim)
    }

    # 返回对应的融合类实例，如果 fusion_type 不在字典中，返回 None
    return fusion_dict.get(fusion_type, None)

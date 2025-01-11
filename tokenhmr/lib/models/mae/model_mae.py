import torch
import torch.nn.functional as F
import tokenhmr.lib.models.mae.models_vit as models_vit
from tokenhmr.lib.models.mae.pos_embed import interpolate_pos_embed
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
# 过滤掉解码器、mask_token 和分类头（head.weight, head.bias）的权重
def load_encoder_weights(checkpoint_model):
    encoder_weights = {k: v for k, v in checkpoint_model.items() if not (k.startswith('decoder') or 'mask_token' in k or 'head.weight' in k or 'head.bias' in k)}
    return encoder_weights

def load_pretrained_mae(modelname='vit_large_patch16',
                        droppath=0.1,
                        globalpool=False,
                        finetune_path=os.path.join(script_dir, 'checkpoint-160.pth')):
    """
    加载预训练的 MAE 编码器权重并返回模型
    """
    # 初始化模型
    model = models_vit.__dict__[modelname](
        drop_path_rate=droppath,
        global_pool=globalpool,
    )

    # 如果提供了 finetune 路径，加载预训练权重
    if finetune_path:
        checkpoint = torch.load(finetune_path, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % finetune_path)
        checkpoint_model = checkpoint['model']

        # 过滤掉解码器、mask_token 和分类头权重
        encoder_weights = load_encoder_weights(checkpoint_model)

        # 插值位置嵌入 (positional embedding)
        interpolate_pos_embed(model, encoder_weights)

        # 加载预训练模型权重
        msg = model.load_state_dict(encoder_weights, strict=False)
        print(msg)  # 输出加载情况，包括 missing_keys 和 unexpected_keys 信息

    return model

# 定义一个简单的测试函数
def test_mae_input_output(model, input_tensor):
    """
    测试 MAE 模型的输入输出结构，并调整输入到合适的尺寸（224x224）

    参数:
    - model: 已加载的 MAE 模型
    - input_tensor: 输入的四维张量 [batch_size, channels, height, width]

    返回:
    - 输出特征的形状
    """
    # 确保输入是四维张量
    assert len(input_tensor.shape) == 4, "输入 tensor 应该是四维 [batch_size, channels, height, width]"

    # 调整输入大小为 MAE 所期望的 [batch_size, channels, 224, 224]
    if input_tensor.shape[2] != 224 or input_tensor.shape[3] != 224:
        input_tensor = F.interpolate(input_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model.forward_features(input_tensor) # 通过MAE编码器
    return output

# 测试程序
if __name__ == "__main__":
    # 加载模型
    print("Help")
    model = load_pretrained_mae(
        modelname='vit_large_patch16',
        droppath=0.1,
        globalpool=False,
        finetune_path='checkpoint-160.pth'  # 替换为实际的checkpoint路径
    )

    # 生成随机的输入张量 [batch_size, channels, height, width]
    # 例如生成一个 batch 大小为 2，通道数为 3，分辨率为 256x256 的随机张量
    random_input = torch.randn(2, 3, 256, 256)  # batch_size=2, channels=3, height=256, width=256

    # 调用测试函数，检查输出
    test_mae_input_output(model, random_input)

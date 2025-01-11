from tokenhmr.lib.models.mae import model_mae
import torch

# 假设我们使用的 MAE 模型输入的是 16x16 的 patch，因此深度图也要满足输入要求

# 定义一个输入的四维张量，假设 batch_size=8, channels=3, height=224, width=224
batch_size = 8
channels = 3  # 假设深度图是三通道的 RGB 图像
height, width = 448, 448  # 输入的图像大小
input_tensor = torch.randn(batch_size, channels, height, width)

print('tag3')
# 加载预训练的 MAE 模型 (假设 finetune_path 为已经训练好的权重文件路径)
mae_model = model_mae.load_pretrained_mae()
print('tag1')
# 调用测试函数，查看输出
output = model_mae.test_mae_input_output(mae_model, input_tensor)
import torch
import cv2
import numpy as np
from mmseg.apis import inference_model, init_model
from tokenhmr.lib.models.utils.crop_new import get_person_mask
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

class DepthModel:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.model = init_model(config=os.path.join(script_dir, 'sapiens_2b_render_people-1024x768.py'),
                    checkpoint=os.path.join(script_dir, 'sapiens_2b_render_people_epoch_25.pth'),
                    device=self.device)
        print("model load asdasdasdasdasdsada")

    def depth_estimation_inference(self, x, flip=False):
        """
        输入为四维张量 batch['img']，输出每个人的深度图并保存到指定目录，并返回四维张量。

        Args:
            batch: 包含 'img' 键的字典，'img' 是 [batch_size, channels, height, width] 的四维张量。
            model: 已初始化的深度估计模型。
            mask_dir: 用于读取实例分割 mask 的目录。
            box_dir: 用于读取 bounding box 的目录。
            output_dir: 输出目录，用于保存深度图。
            flip: 是否使用图像左右翻转进行深度估计增强。

        Returns:
            depth_maps: 四维张量，形状为 [batch_size, 3, height, width]。
        """
        batch_size = x.shape[0]
        depth_maps = []

        for i in range(batch_size):
            # 提取单张图像，并从 Tensor 转为 NumPy 格式
            image = x[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 转为 BGR 格式

            # 创建与原图大小一致的全1 bool矩阵
            mask = torch.ones((x[i].shape[1], x[i].shape[2]), dtype=torch.bool)

            # 推理深度图
            result = inference_model(self.model, image_bgr)
            depth_map = result.pred_depth_map.data.cpu().numpy()[0]  # [H, W]

            # 如果使用翻转增强
            if flip:
                image_flipped = cv2.flip(image_bgr, 1)
                result_flipped = inference_model(model, image_flipped)
                depth_map_flipped = result_flipped.pred_depth_map.data.cpu().numpy()[0]
                depth_map_flipped = cv2.flip(depth_map_flipped, 1)
                depth_map = (depth_map + depth_map_flipped) / 2  # 平均化

            # 只有在 mask 不为 None 的情况下才进行 mask 操作
            mask = self.resize_bool_array(mask, (256, 256))
            mask = mask > 0
            depth_map[~mask] = np.nan  # 只保留 mask 内的深度值

            # 归一化并保留三通道处理
            depth_foreground = depth_map[mask]
            processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)
            if len(depth_foreground) > 0:
                min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
                depth_normalized_foreground = 1 - ((depth_foreground - min_val) / (max_val - min_val))
                depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(np.uint8)
                depth_colored_foreground = cv2.applyColorMap(depth_normalized_foreground, cv2.COLORMAP_INFERNO)
                depth_colored_foreground = np.squeeze(depth_colored_foreground)  # 去掉多余的维度
                processed_depth[mask] = depth_colored_foreground

            # 确保深度图调整为与原始输入相同的尺寸 [H, W]
            depth_image_resized = cv2.resize(processed_depth, (x[i].shape[2], x[i].shape[1]))  # 调整为原始尺寸

            # 将调整后的深度图添加到 depth_maps 列表中
            depth_maps.append(depth_image_resized)  # 每张深度图为 [H, W, 3]

        # 将深度图列表转换为四维张量 [batch_size, 3, height, width]
        depth_maps_tensor = torch.tensor(np.stack(depth_maps)).permute(0, 3, 1, 2).float()  # [batch_size, 3, H, W]

        return depth_maps_tensor

    def resize_bool_array(self, bool_array, target_size, interpolation=cv2.INTER_LINEAR):
        """
        将一个布尔数组等比缩放为指定大小。
        参数:
        - bool_array: 输入的布尔类型二维数组。
        - target_size: 目标大小 (width, height)，即缩放后的宽度和高度。
        - interpolation: OpenCV 插值方法，默认为 cv2.INTER_LINEAR。

        返回:
        - 缩放后的布尔类型二维数组。
        """
        # 将布尔数组转换为 uint8 类型，0 表示 False，1 表示 True
        uint8_array = np.array(bool_array, dtype=np.uint8)
        
        # 使用 OpenCV 进行缩放
        resized_uint8_array = cv2.resize(uint8_array, target_size, interpolation=interpolation)
        
        # 根据需要，将缩放后的数组重新转换为布尔类型
        resized_bool_array = resized_uint8_array > 0  # 大于 0 的元素设置为 True，其它为 False
        
        return resized_bool_array


        
    
if __name__ == "__main__":
    # 测试函数
    def test_depth_estimation_inference():
        # 模拟输入张量，形状为 [batch_size, channels, height, width]
        batch_size = 2
        channels = 3
        height = 768
        width = 1024

        # 创建随机输入数据，值在 [0, 1] 范围内
        x = torch.rand(batch_size, channels, height, width)

        try:
            # 调用深度估计推理函数
            depth_maps_tensor = depth_estimation_inference(x, flip=True)

            # 输出推理结果的形状
            print("输出深度图张量的形状:", depth_maps_tensor.shape)

            # 验证输出张量的形状是否符合预期
            assert depth_maps_tensor.shape == (batch_size, 3, height, width), "输出的深度图形状不正确"
            print("测试通过: 输出深度图形状正确")

        except Exception as e:
            print(f"测试失败: {e}")

    # 运行测试函数
    test_depth_estimation_inference()

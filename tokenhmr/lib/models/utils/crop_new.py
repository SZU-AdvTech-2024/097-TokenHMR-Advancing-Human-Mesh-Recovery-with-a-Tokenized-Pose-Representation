import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 加载YOLO分割模型
segmentation_model = YOLO("yolo11n-seg.pt")  # 使用YOLOv8 segmentation模型

show = False

def get_person_mask(image):
    """
    输入一张图片，返回检测到的人的mask。

    Args:
        image: 输入图像，格式为BGR（OpenCV读取的图像格式），形状为[H, W, C]。

    Returns:
        mask: 二维 NumPy 数组，表示检测到的人的 mask。若未检测到人，则返回 None。
    """
    try:
        # 使用分割模型直接在整张图片上进行实例分割
        segmentation_results = segmentation_model.predict(image, show=False)
        masks = segmentation_results[0].masks  # 获取分割结果的 masks

        # 如果检测到了 mask，并且存在数据
        if masks is not None and masks.data.shape[0] > 0:
            # 获取分割后的第一个mask，并确保其为二维
            mask = masks.data[0].cpu().numpy()  # 假设只取第一个对象的 mask
            if mask.ndim == 2:  # 确保 mask 是二维的
                # 将 mask 转换为 uint8 格式，值范围 [0, 255]
                final_mask = (mask * 255).astype(np.uint8)
                if(show):
                    plt.imshow(final_mask, cmap='gray')
                    plt.title('Person Mask')
                    plt.axis('off')
                    plt.show()

                return final_mask

        # 如果没有检测到 mask，则返回 None
        return None

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None


#Test
if __name__ == '__main__':
    # 读取测试图片
    image_path = 'test_fig.png'
    image = cv2.imread(image_path)

    # OpenCV 默认读取的图像是 BGR 格式，将其转换为 RGB 格式，适应 matplotlib 的显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取人物的 mask
    final_mask = get_person_mask(image_rgb)

    # 检查是否得到了有效的 mask
    if final_mask is not None:
        # 可视化原始图像和 mask
        plt.figure(figsize=(10, 5))

        # 显示 mask
        plt.imshow(final_mask, cmap='gray')
        plt.title('Person Mask')
        plt.axis('off')

        # 展示图片
        plt.show()
    else:
        print("No person detected or mask not found.")
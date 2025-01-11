import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 加载YOLO检测模型
detection_model = YOLO("yolo11n.pt")  # 使用YOLOv8检测模型

# 加载YOLO分割模型
segmentation_model = YOLO("yolo11n-seg.pt")  # 使用YOLOv8 segmentation模型

show = False

def get_person_mask_and_bbox(image):
    """
    输入一张图片，返回检测到的人的mask和bbox坐标。

    Args:
        image: 输入图像，格式为BGR（OpenCV读取的图像格式），形状为[H, W, C]。

    Returns:
        mask: 二维 NumPy 数组，表示检测到的人的 mask。若未检测到人，则返回 None。
    """
    try:

        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()
        # 使用YOLO模型进行人体检测

        detection_results = detection_model.predict(image, show=False)
        boxes = detection_results[0].boxes.xyxy.cpu().tolist()
        clss = detection_results[0].boxes.cls.cpu().tolist()

        # 遍历检测结果，寻找第一个 "person" 类别
        for box, cls in zip(boxes, clss):
            if detection_model.names[int(cls)] == "person":
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box)

                # 提取框选区域（裁剪人物）
                crop_obj = image[y1:y2, x1:x2]
                # 对裁剪的区域进行实例分割
                segmentation_results = segmentation_model.predict(crop_obj, show=False)
                masks = segmentation_results[0].masks

                if masks is not None and masks.data.shape[0] > 0:
                    # 获取分割后的mask并确保其为二维
                    mask = masks.data[0].cpu().numpy()  # 选择第一个mask（假设只有一个对象）
                    if mask.ndim == 2:  # 确保mask是二维
                        # 调整mask尺寸以匹配裁剪框的大小
                        mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))

                        # 创建一个与原图大小一致的空白mask
                        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        final_mask[y1:y2, x1:x2] = (mask_resized * 255).astype(np.uint8)

                        # 显示 mask
                        if(show):
                            plt.imshow(final_mask, cmap='gray')
                            plt.title('Person Mask')
                            plt.axis('off')
                            plt.show()

                        # 返回mask和边界框
                        return final_mask

        # 如果未检测到 "person" 类别，返回 None
        return None

    except Exception as e:
        print(f"crop.py: Error processing image: {str(e)}")
        return None

#Test
if __name__ == '__main__':
    # 读取测试图片
    image_path = 'test_fig.png'
    image = cv2.imread(image_path)

    # OpenCV 默认读取的图像是 BGR 格式，将其转换为 RGB 格式，适应 matplotlib 的显示
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取人物的 mask
    final_mask = get_person_mask_and_bbox(image_rgb)

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
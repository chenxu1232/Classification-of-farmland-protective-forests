# 八通道整体
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        # 1. 读取 8 通道数据
        image_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".npy")
        try:
            jpg = np.load(image_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据文件: {image_path}")
        
        # 此时 jpg 的范围是 0 - 5800 左右，类型 uint16
        jpg = jpg.astype(np.float32)

        # ---------------------------------------------------
        # 【核心修复】归一化 (Normalization)
        # ---------------------------------------------------
        # 既然最大值在 5800 左右，这是典型的反射率数据 (scale=10000)
        # 我们将其除以 10000，将其映射到 0.0 - 0.6 左右的范围，这对神经网络非常健康。
        jpg = jpg / 10000.0
        
        # 加上 clip 防止极少数噪点超过 1.0 (虽然你的最大值才5800，但这行代码很安全)
        jpg = np.clip(jpg, 0, 1.0)

        # ---------------------------------------------------
        # 2. 读取 标签数据
        # ---------------------------------------------------
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")
        png = Image.open(png_path)
        
        # ... (后续的数据增强、transpose 代码保持不变) ...
        
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)
        
        jpg = np.transpose(jpg, [2, 0, 1])
        png = np.array(png)
        
        # 如果是二分类，确保 label 没问题
        if self.num_classes == 2:
             png[png >= self.num_classes] = 0
        else:
             png[png >= self.num_classes] = self.num_classes

        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, random=True):
        """
        针对 8 通道数据的增强，移除了 HSV 变换
        """
        if isinstance(label, Image.Image):
            label = np.array(label)

        iw, ih  = image.shape[1], image.shape[0]
        h, w    = input_shape

        if not random:
            # 验证集：简单的 Resize 和 Padding，保持长宽比
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

            # 填充背景值：图像填 0 或 128 均可，这里保持一致性填 0 (假设已归一化)
            # 如果未归一化且是uint8，可以填128。如果是float数据，建议填0。
            # 这里为了保险，填 0。
            new_image = np.full((h, w, image.shape[2]), 0, dtype=np.float32)
            new_label = np.full((h, w), 0, dtype=np.uint8) # 标签背景类通常是0

            dx = (w-nw)//2
            dy = (h-nh)//2
            
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            new_label[dy:dy+nh, dx:dx+nw] = label

            return new_image, new_label

        # ==========================================================
        # 训练集数据增强
        # ==========================================================
        
        # 1. Resize and jitter (长宽比抖动 + 缩放)
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.5, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        
        # cv2.resize 在高版本 OpenCV 支持多通道，一般无需拆分
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (nw, nh), interpolation=cv2.INTER_NEAREST)

        # 2. Flip (翻转)
        flip = self.rand()<.5
        if flip: 
            image = cv2.flip(image, 1) # 1: 水平翻转
            label = cv2.flip(label, 1)

        # 3. Place on canvas (Padding/Crop)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        
        new_image = np.full((h, w, image.shape[2]), 0, dtype=np.float32)
        new_label = np.full((h, w), 0, dtype=np.uint8)
        
        # 安全赋值，处理 crop 越界情况
        if nw > w or nh > h:
             # 如果 resize 后比目标输入大，直接 resize 到目标输入
             image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
             label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
             new_image = image
             new_label = label
        else:
            # 否则贴图
            new_image[dy:dy+nh, dx:dx+nw, :] = image
            new_label[dy:dy+nh, dx:dx+nw] = label

        image = new_image
        label = new_label

        # 4. Blur (高斯模糊)
        # 【修复】cv2.GaussianBlur 有时不支持 >4 通道，改为分通道处理更稳健
        blur = self.rand() < 0.25
        if blur: 
            if image.shape[2] > 4:
                channels = cv2.split(image)
                blurred_channels = []
                for ch in channels:
                    ch_blur = cv2.GaussianBlur(ch, (5, 5), 0)
                    blurred_channels.append(ch_blur)
                image = cv2.merge(blurred_channels)
            else:
                image = cv2.GaussianBlur(image, (5, 5), 0)

        # 5. Rotate (旋转)
        rotate = self.rand() < 0.25
        if rotate: 
            center      = (w // 2, h // 2)
            rotation    = np.random.randint(-10, 11)
            # 获取旋转矩阵
            M           = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            
            # --- 核心修改：分通道处理旋转 ---
            # warpAffine 的 borderValue 参数不支持 >4 个数值的 tuple
            if image.shape[2] > 4:
                channels = cv2.split(image)
                warped_channels = []
                for ch in channels:
                    # 单通道 warp，borderValue 填 0 (假设 float32 且已归一化)
                    warped_ch = cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)
                    warped_channels.append(warped_ch)
                image = cv2.merge(warped_channels)
            else:
                # 4通道及以下可以直接处理
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0,)*image.shape[2])
            
            # 标签处理
            label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

        return image, label


def seg_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
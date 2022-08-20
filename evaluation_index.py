import torch
import numpy as np
import cv2

class ConfusionMatrix():
    """
    根据混淆矩阵求准确率，mIoU
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (target >= 0) & (target < n)

            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu




class boundary_iou():
    """
    计算boundary iou
    """

    def __init__(self):
        self.intersection = 0
        self.union = 0

    def update(self, target, output):
        #将数据由tensor转numpy，便于后续计算
        target = target.cpu().numpy()
        output = output.cpu().numpy()

        for i in range(len(target)):
            intersection, union = self.boundary_iou(target[i], output[i])
            self.intersection+=intersection
            self.union += union

    def compute(self):
        # 根据边界交集和并集计算boudary iou
        b_iou = self.intersection/self.union
        return b_iou

    def boundary_iou(self, gt, dt, dilation_ratio=0.02):

        gt = gt.astype(np.uint8)
        dt = dt.astype(np.uint8)

        #计算预测图片的边界和GT图片的边界的交并集
        gt_boundary = self.mask_to_boundary(gt, dilation_ratio)
        dt_boundary = self.mask_to_boundary(dt, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        return intersection, union

    def mask_to_boundary(self, mask, dilation_ratio=0.02):
        h, w = mask.shape
        # 计算图像对角线长度，与dilation_ratio相乘得到腐蚀卷积核大小
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        mask = mask.astype(np.uint8)
        # 给原图的四周添加0, 这样连通边界区域的目标像素也会被腐蚀掉
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        #腐蚀操作
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        # 因为之前向四周填充了0, 故而这里不再需要四周
        mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
        # 返回边界
        return mask - mask_erode





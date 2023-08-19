import json
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import create_feature_extractor


# from VGG import get_vgg

def get_vgg(depth=16):
    vgg = getattr(torchvision.models, "".join(['vgg', str(depth)]))()
    vgg.avgpool.output_size = (1, 1)
    vgg.classifier = nn.Linear(512, 100)

    return vgg


def getCAM(img, layer_name, weights, cls_idx):
    """
    获取类别激活映射图
    :param img: 模型输入
    :param layer_name: 提取特征图的节点名称
    :param weights: 全连接层权重
    :param cls_idx: 类别索引
    :return: 类别激活映射图
    """
    # 获取对应类别的权重
    cls_weights = weights[cls_idx].detach().unsqueeze(0)

    # 特征图提取
    feature_extractor = create_feature_extractor(model, return_nodes={layer_name: "feature_map"})
    forward = feature_extractor(img)
    b, c, h, w = forward["feature_map"].shape
    feature_map = forward["feature_map"].detach().reshape(c, h * w)

    # 计算CAM
    # 激活类别特征映射
    CAM = torch.mm(cls_weights, feature_map).reshape(h, w)

    # 归一化后映射到0-255
    CAM = (CAM-torch.min(CAM)) / (torch.max(CAM)-torch.min(CAM))
    CAM = (CAM.numpy() * 255).astype("uint8")

    return CAM


if __name__ == '__main__':
    # 获取类别标签
    # VGG适用于100分类
    with open("class_label/label_100.json") as f:
        label = json.load(f)

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = get_vgg(16)
    model.load_state_dict(torch.load("vgg16_100.pth"))
    model.eval()

    # 全连接层的权重
    last_layer = list(model.modules())[-1]
    fc_weights = last_layer.weight

    img_path = "ipod.jpg"
    original_img = Image.open(img_path)

    # softmax计算概率
    img = transform(original_img).unsqueeze(0)
    output = model(img)
    psort = torch.sort(F.softmax(output, dim=1), descending=True)
    prob, cls_idx = psort

    # top5的类别和概率
    top5 = [(i.item(), j.item()) for i, j in zip(cls_idx.view(-1), prob.view(-1))][:5]

    fig, axs = plt.subplots(2, 3)
    axs.reshape(-1)[0].imshow(np.asarray(original_img))

    for idx, cls_prob in enumerate(top5):
        # 修改layer_name可以获取不同层的CAM
        CAM = getCAM(img, "features", fc_weights, cls_prob[0])

        # 上采样到原图大小
        upsample = cv2.resize(CAM, original_img.size)

        # 热力图
        heatmap = cv2.applyColorMap(upsample, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        result = heatmap * 0.6 + np.asarray(original_img) * 0.4

        axs.reshape(-1)[idx + 1].imshow(np.uint8(result))
        axs.reshape(-1)[idx + 1].text(-10, -10, f"{label[str(cls_prob[0])][1]}: {cls_prob[1]:.3f}", fontsize=12,
                                      color="black")
    plt.show()

import torch
import torchvision
import torchvision.transforms as transforms

import transforms as ext_transforms
from collections import OrderedDict
import utils

from PIL import Image

import numpy as np
import cv2

import matplotlib
import matplotlib.pyplot as plt

import time

from models.enet import ENet
from args import get_arguments
from models.pointrend import PointRend, PointHead
from args import USE_POINT_REND

args = get_arguments()
args.device = args.device if torch.cuda.is_available() else 'cpu'
device = torch.device(args.device)

color_encoding = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('road', (128, 64, 128)),
    ('sidewalk', (244, 35, 232)),
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('pole', (153, 153, 153)),
    ('traffic_light', (250, 170, 30)),
    ('traffic_sign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)),
    ('rider', (255, 0, 0)),
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)),
    ('bus', (0, 60, 100)),
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)),
    ('bicycle', (119, 11, 32))
])

class_name = ['unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign',
              'vegetation',
              'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

if __name__ == '__main__':
    matplotlib.use('TKAgg')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.style.use(['fast'])
    plt.rcParams['figure.facecolor'] = 'gray'

    if not USE_POINT_REND:
        model = ENet(num_classes=20).to(device)
        checkpoint = torch.load('./save/ENet_Cityscapes/ENet.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        num_cla = 20  # 分类
        in_c_fm = 64  # PointHead特征输入层的通道数
        model = PointRend(ENet(num_classes=num_cla), PointHead(in_c=num_cla + in_c_fm, num_classes=num_cla)).to(device)
        checkpoint = torch.load('./save/ENet_Cityscapes/ENet_pointrend.pth', map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

    ONNX_EXPORT = False
    if ONNX_EXPORT:  # 主要尝试ncnn移动端部署
        img = torch.zeros((1, 3, 512, 512))
        f = './save/' + 'ENet.onnx'
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['input'],
                          output_names=['output'])
    else:
        src_path = r'./images/demo1.jpg'
        # src_path = r'G:/car2.mp4'
        image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        if src_path.endswith('.jpg'):
            image = Image.open(src_path)
            image = image_transform(image)
            image = image.unsqueeze(0)
            print(image.size())
            with torch.no_grad():
                start = time.time()
                predictions = model(image)
                print('Out shape: {} Predict: {:.3f}s'.format(predictions.shape, time.time() - start))

                class_encoding = color_encoding
                label_to_rgb = transforms.Compose([
                    ext_transforms.LongTensorToRGBPIL(class_encoding),
                    transforms.ToTensor()
                ])

                _, predictionx = torch.max(predictions.detach(), 1)

                transf_slices = [label_to_rgb(tensor) for tensor in torch.unbind(predictionx.cpu())]
                color_predictions = torch.stack(transf_slices)

                images = torchvision.utils.make_grid(image.detach().cpu()).numpy()
                labels = torchvision.utils.make_grid(color_predictions).numpy()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
                ax1.imshow(np.transpose(images, (1, 2, 0)))
                ax2.imshow(np.transpose(labels, (1, 2, 0)))
                plt.show()

                # plt.imshow(predictions.squeeze(0)[3, ...])
                # plt.show()
                plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, hspace=0.25, bottom=0.01, top=0.95)
                preds = predictions.squeeze(0)
                for index in range(preds.shape[0]):
                    plt.subplot(4, 5, index + 1)
                    plt.yticks([])
                    plt.xticks([])
                    plt.title(class_name[index])
                    plt.imshow(preds[index, ...])
                plt.show()
        else:
            class_encoding = color_encoding
            label_to_rgb = transforms.Compose([
                ext_transforms.LongTensorToRGBPIL(class_encoding),
                transforms.ToTensor()
            ])
            cap = cv2.VideoCapture(src_path)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    plt.ion()
                    start = time.time()
                    image = image_transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    image = image.unsqueeze(0)
                    predictions = model(image)
                    print('Out shape: {} Predict: {:.3f}s'.format(predictions.shape, time.time() - start))

                    _, predictionx = torch.max(predictions.detach(), 1)

                    transf_slices = [label_to_rgb(tensor) for tensor in torch.unbind(predictionx.cpu())]
                    color_predictions = torch.stack(transf_slices)

                    images = torchvision.utils.make_grid(image.detach().cpu()).numpy()
                    labels = torchvision.utils.make_grid(color_predictions).numpy()
                    # ax1.imshow(np.transpose(images, (1, 2, 0)))
                    # ax2.imshow(np.transpose(labels, (1, 2, 0)))
                    # plt.pause(0.0001)
                    # plt.ioff()
                    cv2.imshow('predict', np.transpose(labels, (1, 2, 0)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # plt.subplots_adjust(left=0.01, right=0.99, wspace=0.05, hspace=0.05, bottom=0.01, top=0.95)
                    # preds = predictions.squeeze(0)
                    # for index in range(preds.shape[0]):
                    #     plt.subplot(4, 5, index + 1)
                    #     plt.yticks([])
                    #     plt.xticks([])
                    #     plt.title(class_name[index])
                    #     plt.imshow(preds[index, ...])
                    # plt.show()

            cap.release()
            cv2.destroyAllWindows()

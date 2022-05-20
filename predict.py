import cv2
import numpy as np
import torch

from net.RISTDnet_model import RISTDnet
from utils.DataLoad import DataLoader

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu

    @torch.no_grad()
    def main():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络
        net = RISTDnet()
        net.load_state_dict(torch.load('model/best_model-232-442.9285888671875.pth', map_location=device))
        net.to(device=device)
        # 数据准备
        test_dataset = DataLoader("data/train/")
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=8,
                                                  shuffle=True)

        image = cv2.imread('data/train/img/111.png', 0)
        tg_label = cv2.imread('data/train/label/111.png', 0)
        tg_label = cv2.threshold(tg_label, 128, 255, type=cv2.THRESH_BINARY)[1] / 255
        bg_label = 1 - tg_label
        cv2.imshow('tg', tg_label)
        cv2.imshow('bg', bg_label)

        cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('img', image)
        cv2.waitKey(0)
        image = torch.tensor(image)
        img = image.to(device=device, dtype=torch.float32)
        img = torch.unsqueeze(torch.unsqueeze(img, dim=0), dim=0)
        # gt_bg = bg_label[0].to(device=device, dtype=torch.float32)
        # gt_tg = tg_label[1].to(device=device, dtype=torch.float32)
        pre = net(img)
        bg = pre[0] * 255
        tg = pre[1] * 255

        bg = torch.squeeze(bg).data.cpu().numpy()
        tg = torch.squeeze(tg).data.cpu().numpy()
        tg = cv2.threshold(tg, 0.1, 1, type=cv2.THRESH_BINARY)[1]
        cv2.imwrite('bg.jpg', np.array(bg))
        cv2.imwrite('tg.jpg', np.array(tg))


    main()

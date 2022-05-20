import torch
import cv2
import glob


def MSE():
    gt_bg = torch.tensor([[1, 1], [0, 1]])
    gt_tg = torch.tensor([[0, 0], [1, 0]])
    pre_bg = torch.tensor([[1, 1], [0, 1]])
    pre_tg = torch.tensor([[0, 0], [1, 0]])
    loss = 0

    bg_square = torch.square(gt_bg - pre_bg)

    tg_sq = torch.square(gt_bg - pre_bg)

    loss = bg_square + tg_sq
    loss = torch.sum(loss)
    pass


def resize_img():
    tRootPath = 'data/val/img/'
    tImgPath = glob.glob(tRootPath + '*.png')
    for pImg in tImgPath:
        img = cv2.imread(pImg, 0)
        img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(pImg, img)
        pass


resize_img()

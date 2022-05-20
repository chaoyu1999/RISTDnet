import torch


def MSELoss(gt, pre):
    """
    :param gt:真值 gt[0]:背景真值     gt[1]:目标真值
    :param pre:预测结果 pre[0]：背景似然图    pre[1]:目标似然图
    :return: loss值
    """
    K = gt[0].shape[0]  # batch_size
    loss = torch.sum(torch.square(gt[1] - pre)) / K
    return loss

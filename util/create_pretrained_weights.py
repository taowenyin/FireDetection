import torch


if __name__ == '__main__':
    pretrained_weights = torch.load('../checkpoint/detr-r50-e632da11.pth')

    num_class = 2  # 这里是你的物体数+1，因为背景也算一个
    pretrained_weights["model"]["class_embed.weight"].resize_(num_class + 1, 256)
    pretrained_weights["model"]["class_embed.bias"].resize_(num_class + 1)
    torch.save(pretrained_weights, "../checkpoint/detr-r50_%d.pth" % num_class)

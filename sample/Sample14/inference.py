import matplotlib
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from model.UnDICnet import UnDICnet_s

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
@torch.no_grad()
def main():
    img1_path = r"Sample1.tif"
    img2_path = r"Sample2.tif"

    img_pairs = zip([img1_path], [img2_path])
    network_data = torch.load('../../result/UnDIC-Net.pth')
    model = UnDICnet_s(args=None)
    model.load_state_dict(network_data)
    model = model.to(device)
    model.eval()
    for (img1_file, img2_file) in img_pairs:
        img1 = cv.imread(img1_file, 0)  # 加载为灰度图像
        img2 = cv.imread(img2_file, 0)  # 加载为灰度图像
        img1 = img1/255.0
        img2 = img2/255.0
        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
        img1 = img1.to(device)
        img2 = img2.to(device)

    output = model(img1,img2).cpu()
    disp_x =  output[0, 0, :, :]
    norm = matplotlib.colors.Normalize(vmin=-0.1, vmax=0.1)
    plt.imshow(disp_x, cmap=plt.cm.bwr,norm=norm)
    plt.colorbar()
    np.savetxt(r'Sample14.csv', disp_x, delimiter=',')
    plt.show()
if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms as transforms 
from PIL import Image 



def double_conv(in_c, out_c): #blue horizontal arrows 
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv 

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64) #input one channel 
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        # up_trans' = green arrows; up_conv's = blue arrow 
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.up_conv_4 = double_conv(128, 64)

        #output layer
        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=1
        )

    
    def forward(self, image):
        # bs, c, h, w 
        # encoder
        x1 = self.down_conv_1(image) #to be cropped and concatenated 
        #print(x1.size())
        x2 = self.max_pool_2x2(x1)
        x3= self.down_conv_2(x2) #to be cropped and concatenated
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4) #to be cropped and concatenated
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6) #to be cropped and concatenated
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        #print(x9.size())

        #decoder
        x = self.up_trans_1(x9)
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))

        x = self.up_trans_2(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x, y], 1))

        x = self.up_trans_3(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x, y], 1))

        x = self.up_trans_4(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x, y], 1))
        #print(x.size())

        x = self.out(x)
        print(x.size())

        #imgplot = plt.imshow(x)
        #plt.show()

        
        return x

if __name__ == "__main__":
    #image = torch.rand((1, 1, 2048, 1400))
    #image = Image.open('1.png') 
    #convert_tensor = transforms.ToTensor()
    #img = convert_tensor(image)
    #img = img[None,:] #adding one dimension 
    


    #print(img.size())
    #print(img)
 
    #imgplot = plt.imshow(img) #2 versus 4 dimensions, torch.tensor.read
    #plt.show()
    #import pdb #debugging 
    #pdb.set_trace()

    #potentially add a layer in each structure

    sample_image = torch.rand(1,1,572,572) #batch_size = 1, channel = 1, width = height = 572
    print(sample_image.size())

    model = UNet()
    print(model(sample_image))

 

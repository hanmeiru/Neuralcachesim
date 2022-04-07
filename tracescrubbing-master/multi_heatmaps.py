from pickletools import uint8
from genheatmap import gen_heatmap_from_path
import os
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import cv2
import numpy as np 

#for each trace file in /Users/hanmeiru/Desktop/research/traces, generate heatmap full & miss, 
#put full in /Users/hanmeiru/Desktop/research/masters/tracescrubbing-master/data/train_images
#put miss in /Users/hanmeiru/Desktop/research/masters/tracescrubbing-master/data/train_masks 



train_path = "/Users/hanmeiru/Desktop/research/masters/tracescrubbing-master/data/train_images"
mask_path = "/Users/hanmeiru/Desktop/research/masters/tracescrubbing-master/data/train_masks"
trace_folder = "/Users/hanmeiru/Desktop/research/RollingTraces"

for i in os.listdir(trace_folder):
     if (i == ".DS_Store"):
         continue
     print(i)
     plt_name = os.path.basename(i) + ".png" 
     in_path_name =  trace_folder + "/" + i
     #print(in_path_name)
     #put full heatmaps in train_images
     full_map = gen_heatmap_from_path(in_path_name, 100, "full")
     plt.rcParams.update({'font.size': 10})
     plt.figure(figsize=(10,10))
     plt.imshow(full_map, cmap='Greys')
     plt.box(False)
     plt.axis("off")
     plt.savefig(train_path + "/" + plt_name, bbox_inches='tight', pad_inches = 0)
     img = Image.open(train_path + "/" + plt_name)

    #  #flip vertically and save as new images 
    #  vflipimg = img.rotate(180)
    #  vflipimg.save(train_path + "/" + os.path.basename(i) + "vf.png")
    #  #flip to side and save as new images
    #  sflipimg = img.rotate(90)
    #  sflipimg.save(train_path + "/" + os.path.basename(i) + "sf.png")
    #  #flip horizontally and save as new images
    #  hflipimg = PIL.ImageOps.mirror(img)
    #  hflipimg.save(train_path + "/" + os.path.basename(i) + "hf.png")



     #put miss heatmaps in train_masks
     miss_map = gen_heatmap_from_path(in_path_name, 100, "miss")
     plt.rcParams.update({'font.size': 10})
     plt.figure(figsize=(10,10))
     plt.imshow(miss_map, cmap='Greys')
     plt.box(False)
     plt.axis("off")
     plt.savefig(mask_path + "/" + plt_name, bbox_inches='tight', pad_inches = 0) 

     #invert color of the masks 
     image = Image.open(mask_path + "/" + plt_name)
     inverted_image = PIL.ImageOps.invert(image.convert('L'))
     #inverted_image.save(mask_path + "/" + plt_name)

     #convert to binary 
     # Convert Image to Numpy as array 
     img = np.array(inverted_image)  
     # Put threshold to make it binary
     binarr = np.where(img<1, 0, 1)
     # Covert numpy array back to image 
     binimg = Image.fromarray((binarr*255).astype(np.uint8))
     binimg.save(mask_path + "/" + plt_name)

    #  #flip vertically and save as new images 
    #  vflipimg = binimg.rotate(180)
    #  vflipimg.save(mask_path + "/" + os.path.basename(i) + "vf.png" )
    #  #flip to side and save as new images
    #  sflipimg = binimg.rotate(90)
    #  sflipimg.save(mask_path + "/" + os.path.basename(i) + "sf.png")
    #  #flip horizontally and save as new images
    #  hflipimg = PIL.ImageOps.mirror(binimg)
    #  hflipimg.save(mask_path + "/" + os.path.basename(i) + "hf.png")



 #can put a few in the validation folder (by hand for now)


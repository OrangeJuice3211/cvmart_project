import os

path = '/project/train/src_repo/dataset/kuaidi/'
if not os.path.isdir(path):
    os.mkdir(path)

# train segmentation
txtName = '/project/train/src_repo/dataset/kuaidi/Training_seg.txt'
f = open(txtName, 'w+')

path = "/home/data/305/"
path_mask = "/home/data/305/"

path_list = os.listdir(path)
img_list=[]
for name in path_list:
    if name[-4::] == '.jpg':
        img_list.append(name)
# path_list_mask = os.listdir(path_mask)
# mask_list=[]
# for name in path_list_mask:
#     if name[-4:] == '.png':
#         mask_list.append(name)
img_list.sort()
k=1
lenth = int(len(img_list) / 5)
image_names_val = img_list[k * lenth:(k + 1) * lenth]
image_names_train = [i for i in img_list if i not in image_names_val]
mask_names_train=[]
mask_names_val=[]
for file in image_names_train:
    mask_names_train.append(os.path.splitext(file)[0] + '.png') 
for file in image_names_val:
    mask_names_val.append(os.path.splitext(file)[0] + '.png') 
# mask_list.sort()

for i in range(len(image_names_train)):
    trainIMG = image_names_train[i]
    trainGT = mask_names_train[i]
    result = trainIMG + ' ' + trainGT +'\n'
    f.write(result)

f.close()

# val segmentation
txtName = '/project/train/src_repo/dataset/kuaidi/Validation_seg.txt'
f = open(txtName, 'w+')

for i in range(len(image_names_val)):
    trainIMG = image_names_val[i]
    trainGT = mask_names_val[i]
    result = trainIMG + ' ' + trainGT +'\n'
    f.write(result)

f.close()

# # test segmentation
# txtName = 'ISIC/Testing_seg.txt'
# f = open(txtName, 'a+')
#
# path = "../seg_data/ISIC-2018_Testing_Data/Images/"
# path_mask = "../seg_data/ISIC-2018_Testing_Data/Images/"
# # path_mask = "../seg_data/ISIC-2018_Testing_Data/Annotation/"
#
# path_list = os.listdir(path)
# path_list_mask = os.listdir(path_mask)
# path_list.sort()
# path_list_mask.sort()
#
# for i in range(len(path_list)):
#     trainIMG = path[-7::]+path_list[i]
#     trainGT = path_mask[-11::]+path_list_mask[i]
#     result = trainIMG + ' ' + trainGT +'\n'
#     f.write(result)
#
# f.close()

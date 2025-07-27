from matplotlib import pyplot
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
#import Image
from PIL import Image
import cv2
import random
import collections
import tqdm
from tqdm.notebook import tqdm_notebook

dataDir = 'D:/BaiduNetdiskDownload/coco'
dataType = 'train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# initialize COCO api for instance annotations
coco=COCO(annFile)

# 旋转和缩放
rotate_rate = 0.2       # 一张图片用旋转的概率
scaling_rate = 0.2   # 一张图使用缩放的概率


rotate_angle = (-30,30)   # 旋转角度
scaling_factor = (0.6,1.3)  # False or (0.5,1.5)







# imgdata为cv.read()的返回,lowerCutoff为最小目标区域面积，upperCutoff为最大目标区域面积(copy move有两个区域，所以最后获得原区域和篡改区域的总面积为这个数字乘2)
# 返回为mask
def findAnnotationMatchingCriteria(anns, imgdata, lowerCutoff=0.01, upperCutoff=0.6):
    annotation_count = len(anns)   # 这张图有多少个已标注的目标
    toReturn = []
    indexes = random.sample(range(annotation_count), annotation_count)  # 随机把这些目标乱序，得到列表

    # 遍历这个列表，把第一个满足要求的目标弹出，作为结果返回
    while len(indexes) > 0:
        h, w, c = imgdata.shape
        mask = coco.annToMask(anns[indexes.pop()])
        # area = anns[indexes.pop()]['area']


        forgedPixelsCount = collections.Counter(mask.flatten())[1]
        # hold = forgedPixelsCount / (h * w)
        # print(forgedPixelsCount, 3333)
        if forgedPixelsCount >= (h * w * lowerCutoff) and forgedPixelsCount <= (h * w * upperCutoff):
            toReturn = mask
            break

    return toReturn


# 输入的两者一个是前景rgb；一个是灰度图，只有前景值为1；两者形状一样
def getAffineTransformedMask_2(foreground, binarymask):
    indices = np.where(binarymask == 1)

    # 物体的四个边角点坐标
    upper = np.min(indices[0])
    lower = np.max(indices[0])
    left = np.min(indices[1])
    right = np.max(indices[1])

    # 物体的宽和高
    width = right - left
    height = lower - upper


    # 判断把篡改区域粘贴到左边还是右边
    n = random.randint(10, 30)
    hor_right = False if (binarymask.shape[1] - (right + n + width)) <= 0 else True
    hor_left = False if (left - (n + width)) <= 0 else True

    side = ""
    if hor_right == True and hor_left == True:
        side = random.sample(["R", "L"], 1)[0]

    elif hor_right == True and hor_left == False:
        side = "R"

    elif hor_right == False and hor_left == True:
        side = "L"
    else:
        return ([], [])

    if side == "L":
        S = -(width + n)
    else:
        S = width + n


    v = 0
    lu = random.randint(0, 1)
    if ((upper - 10) > 1 and (binarymask.shape[0] - lower - 10) > 1):
        if lu == 1:
            v = -random.randint(1, upper - 10)
        else:
            v = random.randint(1, binarymask.shape[0] - lower - 10)
    elif (upper - 10) > 1:
        v = -random.randint(1, upper - 10)
    elif (binarymask.shape[0] - lower - 10) > 1:
        v = random.randint(1, binarymask.shape[0] - lower - 10)
    else:
        return ([], [])

    rows, cols = binarymask.shape
    new_binary_mask = []
    new_foreground = []


    M = np.float32([[1, 0, S], [0, 1, v]])
    transformedForeground = cv2.warpAffine(foreground, M, (cols, rows))
    transformedBinaryMask = cv2.warpAffine(binarymask, M, (cols, rows))

    # print(transformedBinaryMask.shape,binarymask.shape,transformedForeground.shape)


    # 增加旋转和缩放 ==========================================================================================================

    indices = np.where(transformedBinaryMask != 0)
    # 平移后的四个边角点坐标
    upper = np.min(indices[0])
    lower = np.max(indices[0])
    left = np.min(indices[1])
    right = np.max(indices[1])

    center_x = (left+right)/2   # 中心x坐标
    center_y = (upper+lower)/2  # 中心y坐标



    if random.random() < rotate_rate:
        rotate_angle_ = random.randint(rotate_angle[0],rotate_angle[1])
        # rotate_angle_ =  60
    else:
        rotate_angle_ = 0

    if random.random() < scaling_rate:
        scaling_factor_ = random.random()*(scaling_factor[1]-scaling_factor[0]) + (scaling_factor[1]+scaling_factor[0])/2
        # scaling_factor_ = 0.6
    else:
        scaling_factor_ =1


    M = cv2.getRotationMatrix2D((center_x, center_y), rotate_angle_ , scaling_factor_)  # 中间是旋转角度，最后的是缩放系数
    # print(M)
    transformedForeground = cv2.warpAffine(transformedForeground, M, (cols, rows))
    transformedBinaryMask = cv2.warpAffine(transformedBinaryMask, M, (cols, rows))

    
    # 增加旋转 ==========================================================================================================

    # 旋转后再取前景区域
    foreground_1 = transformedForeground.copy()
    foreground_1[:, :, 0] = np.array(transformedForeground[:, :, 0] * transformedBinaryMask)   # 图像和mask相乘，相当于保留原图上目标的区域
    foreground_1[:, :, 1] = np.array(transformedForeground[:, :, 1] * transformedBinaryMask)
    foreground_1[:, :, 2] = np.array(transformedForeground[:, :, 2] * transformedBinaryMask)



    return (foreground_1, transformedBinaryMask)


catIds = coco.getCatIds()
imgIds = coco.getImgIds()


file_name_list = [((coco.loadImgs(imid)[0])['file_name']) for imid in imgIds]
# area = [((coco.loadAnns(imid)[0])['area']) for imid in imgIds]
# print(area)



# 要保存到的文件夹
fake_path = 'D:/dataset/629image/'
# mask_path = 'D:/dataset/tongyong_mask/'
mask_2_path = 'D:/dataset/629mask2/'


path = 'D:/dataset/629_image_/'
path_1 = 'D:/dataset/629_image_s/'
path_2 = 'D:/dataset/629_image_t/'


image_counter = 0       # 40800


for imgId, file_name in tqdm_notebook(zip(imgIds, file_name_list)):

    # if count<= 40800:
    #     count += 1
    #     continue


    # print(imgId, file_name)

    imgdata = cv2.imread('D:/BaiduNetdiskDownload/coco/train2017/' + file_name)
    # Convert to RGB
    b, g, r = cv2.split(imgdata)

    # 排除灰度图像
    if (np.array_equal(np.array(r), np.array(g)) and np.array_equal(np.array(r), np.array(b))):
        print("stop")
        continue  # Exclude gray scale images

    imgdata = cv2.merge([r, g, b])
    # pyplot.imshow(imgdata)
    # pyplot.show()

    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)



    anns = coco.loadAnns(annIds)
    # pyplot.imshow(anns)
    # pyplot.show()
    # print(anns,111)
    binarymask = findAnnotationMatchingCriteria(anns, imgdata)   # 返回mask
    # print(len(binarymask))
    # pyplot.imshow(binarymask)
    # pyplot.show()

    # contours, cnt = cv2.findContours(binarymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # area = cv2.contourArea(contours[0])
    # print(area,111)

    # 如果这张图没有拿到mask，则跳过
    if len(binarymask) == 0:
        continue


    # ##获得篡改后的前景(粘贴目标)区域。
    new_foreground, new_binarymask = getAffineTransformedMask_2(imgdata, binarymask)

    # pyplot.imshow(new_foreground)
    # pyplot.show()
    #
    # pyplot.imshow(new_binarymask)
    # pyplot.show()
    # break
    # #
    # 如果没有目标，则跳过
    if (len(new_foreground) == 0):
        print("stop",image_counter)
        continue

    background = Image.fromarray(imgdata, 'RGB')    # 原图
    # background.save(path_2 + 'image_' + str(image_counter) + '.png')
    new_foreground = Image.fromarray(new_foreground, 'RGB').convert('RGBA')  # 要粘贴的目标区域
    # new_foreground.save(path + 'image_dan_' + str(image_counter) + '.png')

    new_binarymask_2  = new_binarymask

    # binarymask_1 = new_binarymask + binarymask

    new_binarymask_1 = binarymask

    # new_binarymask = new_binarymask
    #
    #
    new_binarymask_2 = Image.fromarray(new_binarymask_2 * 255)

    new_binarymask_1 = Image.fromarray(new_binarymask_1 * 255)
    # binarymask_1 = Image.fromarray(binarymask_1 * 255)



    datas = new_foreground.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((0, 0, 0, 0))
        else:
            newData.append(item)
    new_foreground.putdata(newData)
    # foreground=foreground.resize((background.size[0],background.size[1]),Image.ANTIALIAS)



    background.paste(new_foreground, (0, 0), mask=new_foreground.split()[-1])  # 原图 + 粘贴区域
    # pyplot.imshow(background)
    # pyplot.show()



    # background = background.resize(size=(320, 320))
    # new_binarymask = new_binarymask.resize(size=(320, 320))
    background.save(path + 'image_' + str(image_counter)+'.png')
    #
    #
    #
    #
    new_binarymask_2.save(path_1 + 'mask_'+ str(image_counter)+'.png')
    new_binarymask_1.save(path_2 + 'mask_' + str(image_counter) + '.png')
    # binarymask_1.save(mask_2_path + 'mask_' + str(image_counter) + '.png')

    image_counter += 1

    # if image_counter >=30:
    #     break

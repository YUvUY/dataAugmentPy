import os
from PIL import Image
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage


ia.seed(1)

GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]

def get_inner_bbs(image_path, dst_img_dir, array_info, p_numbers):
    '''
    :param image_path: src img path
    :param dst_img_dir: img save path
    :param coor_array: label coor array
    :param p_numbers: Numbers of images to enhance
    :return: [(bbs_array, img_info),
            (bbs_array, img_info)]
    '''

    try:
    	# 这里将4个坐标值和类别名拆分开，后续再将新的坐标值和标签合为一个数组
        assert array_info.shape[1] == 5
        coor_array = array_info[:, :-1]
        cls_array = array_info[:, -1]

        image = Image.open(image_path)
        image = np.array(image)
        img_name = os.path.split(image_path)[-1].split(".")[0]
        bbs = BoundingBoxesOnImage.from_xyxy_array(coor_array, shape=image.shape)
    except Exception as e:
        print(f"err:{e}")
        print(array_info.shape)
        print(image_path)
        return None

    # # Draw the original picture
    # image_before = draw_bbs(image, bbs, 100)
    # ia.imshow(image_before)

    # Image augmentation sequence
    # 此增强序列可以自行定义，API可以查询imgaug官方文档
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # 水平翻转
        iaa.Flipud(0.5),  # 垂直翻转

        iaa.ChannelShuffle(0.2), # 通道变化

        iaa.Crop(percent=(0, 0.1)),

        iaa.Sometimes( # 0.5的图像做高斯滤波
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),

        # Strengthen or weaken the contrast in each image. 对比度
        iaa.LinearContrast((0.85, 1.2)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

        # change illumination
        iaa.Multiply((0.8, 1.2), per_channel=0.2),

        # affine transformation仿射变化
        iaa.Affine(
            scale={"x": (0.9, 1.5), "y": (0.9, 1.5)}, # 缩放
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # 平移
            rotate=(-45, 45),   # 旋转
            shear=(-8, 8)  # 错切
        )
    ], random_order=True)  # apply augmenters in random order

    res_list = []
    # gen img and coor
    try:
        for epoch in range(p_numbers):
        	# 同时对图片和标签进行变换
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            # 这个方法可以将增强后标签框在图像外部的坐标，变为图片内
            # bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()

            # # draw aug img and label
            image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
            ia.imshow(image_after)

            # save img
            h, w, c = image_aug.shape

            img_aug_name = rf'{dst_img_dir}/{img_name}_{epoch}.jpg'
            im = Image.fromarray(image_aug)
            im.save(img_aug_name)

			# 将新标签和类名合为一个数组，保存到列表
            bbs_array = bbs_aug.to_xyxy_array()
            result_array = np.column_stack((bbs_array, cls_array))
            res_list.append([result_array, (img_aug_name, h, w, c)])
    except Exception as e:
        print(e)
        print(img_aug_name)
        return None
    # return coor and img info
    return res_list

# Pad image with a 1px white and (BY-1)px black border
def _pad(image, by):
    image_border1 = ia.augmenters.size.pad(image, top=1, right=1, bottom=1, left=1,
                           mode="constant", cval=255)
    image_border2 = ia.augmenters.size.pad(image_border1, top=by-1, right=by-1,
                           bottom=by-1, left=by-1,
                           mode="constant", cval=0)
    return image_border2

# Draw BBs on an image
# and before doing that, extend the image plane by BORDER pixels.
# Mark BBs inside the image plane with green color, those partially inside
# with orange and those fully outside with red.
def draw_bbs(image, bbs, border):
    image_border = _pad(image, border)
    for bb in bbs.bounding_boxes:
        if bb.is_fully_within_image(image.shape):
            color = GREEN
        elif bb.is_partly_within_image(image.shape):
            color = ORANGE
        else:
            color = RED
        image_border = bb.shift(x=border, y=border)\
                         .draw_on_image(image_border, size=2, color=color)

    return image_border

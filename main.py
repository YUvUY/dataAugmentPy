from xml.dom.minidom import parse, Document
from imgaug_utils import get_inner_bbs
import numpy as np
import os


def main_change(src_xml_path, src_img_dir, dst_img_dir, dst_xml_dir, p_number):
    '''
    :param p_number: Numbers of images to enhance
    '''
    print(f"src: {src_xml_path}")
    dom = parse(src_xml_path)
    root = dom.documentElement
    img_name = root.getElementsByTagName("path")[0].childNodes[0].data
    img_name = os.path.split(img_name)[-1].split(".")[0]
    img_path = f"{src_img_dir}/{img_name}.png"
    item = root.getElementsByTagName("item")

    # label = root.getElementsByTagName("name")[0].childNodes[0].data

    coor_list = []

    # 便利循环item，获取每个box中包含的信息，保存到coor_list中
    for box in item:
        cls_name = box.getElementsByTagName("name")[0].childNodes[0].data
        x1 = max(0, int(box.getElementsByTagName("xmin")[0].childNodes[0].data))
        y1 = max(0, int(box.getElementsByTagName("ymin")[0].childNodes[0].data))
        x2 = max(0, int(box.getElementsByTagName("xmax")[0].childNodes[0].data))
        y2 = max(0, int(box.getElementsByTagName("ymax")[0].childNodes[0].data))
        cls_name = trans_cls_name(cls_name)
        coor_list.append([x1, y1, x2, y2, cls_name])
    # 将coor_list中的信息，传入get_inner_bbs
    aug_list = get_inner_bbs(img_path, dst_img_dir, np.array(coor_list), p_number)
    if not aug_list:
        return
    # 将返回的坐标以及图片信息传入save_xml函数，保存到对应的xml文件中
    for aug_info in aug_list:
        save_xml(aug_info, dst_xml_dir)


def trans_cls_name(name):
    if name == "align":
        return 0
    elif name == "smoke":
        return 1
    elif name == "drink":
        return 2
    else:
        raise ValueError(f"wrong class name! {name}")


def inv_trans_cls_name(value):
    if value == 0:
        return "align"
    elif value == 1:
        return "smoke"
    elif value == 2:
        return "drink"
    else:
        raise ValueError(f"wrong class number! {value}")


def save_xml(aug_info, dst_xml_dir):
    # 返回的aug_info包含两部分，坐标信息和图片信息，坐标信息的格式是[x1, y1, x2, y2, cls_name]
    coor_array, img_info = aug_info
    # 图片信息中包含增强后保存的图片路径，图片宽，高，通道数
    img_name, img_h, img_w, img_c = list(map(str, img_info))
    xml_name = os.path.split(img_name)[-1].split(".")[0]

    # 1.创建DOM树对象
    dom = Document()
    # 2.创建根节点。每次都要用DOM对象来创建任何节点。
    root_node = dom.createElement('root')
    # 3.用DOM对象添加根节点
    dom.appendChild(root_node)

    path_node = dom.createElement('path')
    root_node.appendChild(path_node)
    path_text = dom.createTextNode(img_name)
    path_node.appendChild(path_text)

    outputs_node = dom.createElement('outputs')
    root_node.appendChild(outputs_node)

    object_node = dom.createElement('object')
    outputs_node.appendChild(object_node)

    # 遍历出每一个item中的坐标和cls name
    for row_data in coor_array:
        cls_name = inv_trans_cls_name(int(row_data[4]))
        item_node = dom.createElement('item')
        object_node.appendChild(item_node)

        name_node = dom.createElement('name')
        item_node.appendChild(name_node)
        name_text = dom.createTextNode(cls_name)
        name_node.appendChild(name_text)

        bndbox_node = dom.createElement('bndbox')
        item_node.appendChild(bndbox_node)

        xmin_node = dom.createElement('xmin')
        bndbox_node.appendChild(xmin_node)
        xmin_text = dom.createTextNode(str(row_data[0]))
        xmin_node.appendChild(xmin_text)

        ymin_node = dom.createElement('ymin')
        bndbox_node.appendChild(ymin_node)
        ymin_text = dom.createTextNode(str(row_data[1]))
        ymin_node.appendChild(ymin_text)

        xmax_node = dom.createElement('xmax')
        bndbox_node.appendChild(xmax_node)
        xmax_text = dom.createTextNode(str(row_data[2]))
        xmax_node.appendChild(xmax_text)

        ymax_node = dom.createElement('ymax')
        bndbox_node.appendChild(ymax_node)
        ymax_text = dom.createTextNode(str(row_data[3]))
        ymax_node.appendChild(ymax_text)

    size_node = dom.createElement('size')
    root_node.appendChild(size_node)

    width_node = dom.createElement('width')
    size_node.appendChild(width_node)
    width_text = dom.createTextNode(img_w)
    width_node.appendChild(width_text)

    height_node = dom.createElement('height')
    size_node.appendChild(height_node)
    height_text = dom.createTextNode(img_h)
    height_node.appendChild(height_text)

    depth_node = dom.createElement('depth')
    size_node.appendChild(depth_node)
    depth_text = dom.createTextNode(img_c)
    depth_node.appendChild(depth_text)

    try:
        with open(rf'{dst_xml_dir}/{xml_name}.xml', 'w') as f:
            # writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，
            # 第三个参数是其他子节点的缩进格式，
            # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
            dom.writexml(f, indent='', addindent='\t', newl='\n', encoding='utf-8')
            print(rf'dst: {dst_xml_dir}/{xml_name}.xml')
    except Exception as err:
        print('错误：{err}'.format(err=err))


if __name__ == '__main__':
    main_change('D:/Proj/chipImg/align/annotations/0001.xml',
                'D:/Proj/chipImg/align/images',
                'D:/Proj/chipImg/align2/augued/images',
                'D:/Proj/chipImg/align2/augued/annotations',
                10)

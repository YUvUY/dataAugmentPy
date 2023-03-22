使用imgaug库进行图像增强，制作数据集，同时生成图像和对应的标签

#### 使用方法

1. 使用精灵标注助手进行原图标注，保存为xml文件，内容如图

   ![image-20230322163512790](https://yuvuy.oss-cn-hangzhou.aliyuncs.com/img/image-20230322163512790.png)

2. 把图片和生成的标注文件分别放入文件夹
3. 创建项目，引入imgaug库（可直接conda insstall imgaug安装，详细步骤见官网）
4. 修改main函数进行生成

#### 主要函数说明

```py
def main_change(src_xml_path, src_img_dir, dst_img_dir, dst_xml_dir, p_number)
# src_xml_path 待增强原图路径(一个jpg或png文件)
# src_img_dir 原图标注路径(一个xml)文件
# dst_img_dir 增强后图片存放路径
# dst_xml_dir 增强后标注存放路径
# p_number 用原图生成几张增强图
# 注：将Line 16末"png"改为自己使用的图片格式
```

```python
def get_inner_bbs(image_path, dst_img_dir, array_info, p_numbers)
# 该函数定义了对图像增强方式，可以自行更改
```



> 参考：https://blog.csdn.net/weixin_42370357/article/details/107677096
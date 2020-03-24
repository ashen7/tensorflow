import tensorflow as tf
#配置文件 定义关于YOLO v1网络的全局变量
# 目标检测数据集的20个类别

voc07_class = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# 20个类别 对应的检测框颜色
colors_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
               (0, 255, 255), (204, 50, 153), (147, 20, 255), (193, 182, 255), (30, 105, 210),
               (220, 245, 245), (0, 215, 255), (212, 255, 127), (0, 255, 127), (238, 238, 175),
               (255, 191, 0), (255, 112, 132), (255, 228, 225), (245, 240, 255), (250, 230, 230)]

# 训练/测试集
voc07_data_path = '/home/yipeng/workspace/darknet/VOCdevkit/VOC2007/'

# 目标检测模型 和预训练的目标分类模型
detect_model_path = '/home/yipeng/workspace/python/tensorflow_samples/model/yolov1/'
pre_trian_model_path = '/home/yipeng/workspace/python/tensorflow_samples/model/yolov1/pretrain/'
log_dir = '/home/yipeng/workspace/python/tensorflow_samples/log/'

# 预训练 用的cifar10 类别是10
pre_train_class_num = 10
epoch = 50
batch_size = 10
learning_rate = 0.001
keep_prob = 0.5
momentum = 0.9

# 网格大小7*7 检测的输入图片大小448*448 每个网格预测2个边界框
cell_size = 7
image_size = 448
per_cell_bbox = 2

# 阈值和检测的比例参数
threshold = 0.1
iou_threshold = 0.5
object_confident_scale = 1.0
no_object_confident_scale = 0.5
coord_scale = 5.0
class_scale = 1.0


batch_norm_params = {
    'decay': 0.998,
    'epsilon': 0.001,
    'scale': False,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
}
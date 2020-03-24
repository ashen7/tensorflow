import cv2 as cv
import numpy as np
import tensorflow as tf
import colorsys

import config as cfg
from yolo import YOLO

class Detector(object):
    def __init__(self, net):
        self.net = net
        self.image_size = cfg.image_size
        self.detect_model_path = cfg.detect_model_path
        self.cell_size = cfg.cell_size
        self.per_cell_bbox = cfg.per_cell_bbox
        self.classes = cfg.voc07_class
        self.num_class = len(self.classes)
        self.threshold = cfg.threshold
        self.iou_threshold = cfg.iou_threshold
        self.bound_box1 = self.cell_size * self.cell_size * self.num_class
        self.bound_box2 = self.bound_box1 + self.cell_size * self.cell_size * self.per_cell_bbox

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=4)

    def draw_result(self, img, result):
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv.rectangle(img, (x - w, y - h), (x + w, y + h), color, 3)
            cv.putText(img, result[i][0], (x - w + 1, y - h + 8), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
            print(result[i][0], ': %.2f%%' % (result[i][5] * 100))

    def detect(self,img):
        h,w,c = img.shape
        img = cv.resize(img, (self.image_size, self.image_size))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.0
        imgs = np.reshape(img, (1,self.image_size,self.image_size,3))

        result = self.detect_from_cvmat(imgs)[0]
        for i in range(len(result)):
            result[i][1] *= (1.0 * w / self.image_size)
            result[i][2] *= (1.0 * h / self.image_size)
            result[i][3] *= (1.0 * w / self.image_size)
            result[i][4] *= (1.0 * h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.session.run(self.net.output, feed_dict={
            self.net.input: inputs, self.net.is_training: False
        })
        print("net_shape:", net_output.shape)
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        print('result: ',results)
        return results

    def img_test(self):
        path = r'/home/yipeng/workspace/python/DL_HotNet_Tensorflow/net/Detection/YOLOV1/test/'
        self.load_model()
        for i in range(1,6):
            im_path = path + str(i) + '.jpg'
            img = cv.imread(im_path)
            result = self.detect(img)
            print(result)
            self.draw_result(img, result)
            cv.imshow(str(i), img)
        cv.waitKey(0)

    def load_model(self):
        model_file = tf.train.latest_checkpoint(self.detect_model_path)
        self.saver.restore(self.session, model_file)

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size, self.per_cell_bbox, self.num_class)) # [7,7,2,20]
        class_probs = np.reshape(output[0:self.bound_box1], (self.cell_size, self.cell_size, self.num_class))    # 类别 [0:7x7x20] - > [7,7,20]
        scales = np.reshape(output[self.bound_box1:self.bound_box2],(self.cell_size, self.cell_size, self.per_cell_bbox))  # [7x7x20:7x7x22] -> [7,7,2]  置信率
        boxes = np.reshape(output[self.bound_box2:], (self.cell_size, self.cell_size, self.per_cell_bbox, 4))   # 坐标 [7x7x22:] -> [7,7,2,4]
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.per_cell_bbox),
                                         [self.per_cell_bbox, self.cell_size, self.cell_size]), (1, 2, 0))  # [7,7,2]

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.per_cell_bbox):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors

    def camera_detector(self, capture, wait=10):
        while (1):
            ret, frame = capture.read()
            result = self.detect(frame)

            self.draw_result(frame, result)
            cv.imshow('Camera', frame)
            cv.waitKey(wait)

            if cv.waitKey(wait) & 0xFF == ord('q'):
                break
        capture.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    yolov1 = YOLO(is_pre_training=False)

    detector = Detector(yolov1)
    # detector.img_test()
    video_capture = cv.VideoCapture(0)
    detector.camera_detector(video_capture)
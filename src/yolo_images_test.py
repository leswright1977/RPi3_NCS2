#Code based on work by PINTO at
#https://github.com/PINTO0309/OpenVINO-YoloV3
#Runs yolov3 on an image in the same directory
#Shows an image containing bounding boxes with labels
#Writes an image containing bounding boxes with labels
#Change lines containng 'imread' and 'imwrite' if You want something other than in.jpg and out.png
import sys, os, cv2, time
import numpy as np, math
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin

m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 80
coords = 4
num = 3
anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]

LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    return (area_of_overlap / area_of_union)


def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects


def main_IE_infer():


    image = cv2.imread('in.jpg')
    image_width = np.size(image,1)
    image_height = np.size(image,0)


    model_xml = "models/frozen_yolo_v3.xml"
    model_bin = "models/frozen_yolo_v3.bin"


    time.sleep(1)

    plugin = IEPlugin("MYRIAD")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)

    prepimg = cv2.resize(image, (m_input_size, m_input_size))
    prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
    prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
    outputs = exec_net.infer(inputs={input_blob: prepimg})

    objects = []

    for output in outputs.values():
        objects = ParseYOLOV3Output(output, m_input_size, m_input_size, image_height, image_width, 0.7, objects)

    # Filtering overlapping boxes
    objlen = len(objects)
    for i in range(objlen):
        if (objects[i].confidence == 0.0):
            continue
        for j in range(i + 1, objlen):
            if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                objects[j].confidence = 0
        
    # Drawing boxes
    for obj in objects:
        if obj.confidence < 0.2:
            continue
        label = obj.class_id
        confidence = obj.confidence
        if confidence > 0.2:
            label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness)
            cv2.rectangle(image, (obj.xmin-1, obj.ymin-1),(obj.xmin+70, obj.ymin-12), (0,0,0), -1)
            cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_text_color, 1)

    while cv2.waitKey(1) < 0:
        cv2.imshow("Result", image)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cv2.imwrite('out.png', image)



cv2.destroyAllWindows()



if __name__ == '__main__':
    sys.exit(main_IE_infer() or 0)


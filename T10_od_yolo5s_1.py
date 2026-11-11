import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from ultralytics.utils import ops
from dx_engine import InferenceEngine, InferenceOption

# def show_img(img, title="Image", size=(10,10)): # Img 확인을 위한 함수
#     plt.figure(figsize=size)
#     if img.ndim == 2:
#         plt.imshow(img, cmap="gray")
#     else:
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

model_path = "./assets/models/YOLOV5S-1.dxnn"
img_path = "./assets/inputs/image1.jpg"

num_anchors = 3
anchors = [
    np.array([[10,13], [16,30], [33,23]]),   
    np.array([[116,90], [156,198], [373,326]]),
    np.array([[30,61], [62,45], [59,119]])  
]
feature_shapes = [(80,80), (20,20), (40,40)]  # 각 scale feature map 크기
strides = [8, 32, 16]

def sigmoid_np(x):
    x = np.clip(x, -88.0, 88.0, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float32))

def yolo_postprocess_numpy(outputs, anchors, strides, num_classes):
    batch = outputs[0].shape[0]
    processed = []

    for idx, output in enumerate(outputs):
        h, w = output.shape[1], output.shape[2]
        stride = np.float32(strides[idx])
        anchor = anchors[idx].astype(np.float32)
        num_anchors = anchor.shape[0]

        output = output.astype(np.float32).reshape(batch, h, w, num_anchors, num_classes + 5)

        output = sigmoid_np(output)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        grid = np.stack((grid_x, grid_y), axis=-1)

        xy = ((output[..., 0:2] * 2.0) - 0.5 + grid[None, ..., None, :]) * stride
        wh = ((output[..., 2:4] * 2.0) ** 2) * anchor[None, None, None, :, :]

        box = np.concatenate((xy, wh), axis=-1)
        final = np.concatenate((box, output[..., 4:]), axis=-1)
        final = final.reshape(batch, -1, num_classes + 5)

        processed.append(final)

    return np.concatenate(processed, axis=1).astype(np.float32)

  # 이미지 위치
frame = cv2.imread(img_path) # 이미지 Load
# show_img(frame, "Original Image") # 이미지 확인

src_shape = frame.shape[:2]
new_shape = (640, 640)

r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])
ratio = r, r
new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
dw /= 2
dh /= 2

resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
image_input = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))

#show_img(image_input, "Letter Boxed Image") # 전처리된 이미지 확인

image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) # 모델에 맞게 RGB로 변환

pad = (dw, dh)

io = InferenceOption()

ie = InferenceEngine(model_path, io)

ie_output = ie.run([image_input]) # 추론 수행


classes = [ # COCO dataset class names
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]
n_classes = len(classes)



if io.get_use_ort()==True:
    final_output = ie_output[0]
    
else:
    ie_output_sliced = [out[..., :255].astype(np.float32) for out in ie_output]
    final_output = yolo_postprocess_numpy(ie_output_sliced, anchors, strides, n_classes)

# print("Raw Output Shape:", [o.shape for o in ie_output[0]])
# # print("Output Shape:", [o.shape for o in final_output])
# # print("Output Sample:", final_output[0][:5])  # 일부 샘플 출력

# --------------------------------------------------------
# YOLOv5s Output: (1, 25200, 85)
# 85 = [x, y, w, h, obj_conf, 80 class_conf]
# --------------------------------------------------------

ie_output0 = final_output[0]  # (25200, 85)

num_det = ie_output0.shape[0]
decoded_tensor = []



for i in range(num_det):
    cx, cy, w, h = ie_output0[i, 0:4]
    
    obj_conf = ie_output0[i, 4]
    class_confs = ie_output0[i, 5:]

    # 가장 높은 클래스 선택
    class_id = np.argmax(class_confs)
    class_score = class_confs[class_id]
    score = obj_conf * class_score



    decoded_tensor.append([cx, cy, w, h, score, class_id])

if len(decoded_tensor) == 0:
    decoded_tensor = np.zeros((1, 6), dtype=float)
else:
    decoded_tensor = np.stack(decoded_tensor)

print("Decoded Tensor Shape:", decoded_tensor.shape)

x = np.squeeze(decoded_tensor)
score_threshold = 0.4 # score 가 0.4 이상인 박스만 존재하도록
iou_threshold = 0.4 # NMS IoU 임계값
colors = np.random.randint(0, 256, [n_classes, 3], np.uint8).tolist() # 클래스별 박스 색상 설정

if x.ndim <= 1 or len(x) == 0:
    print("No detections")


else:
    box = ops.xywh2xyxy(x[..., :4])  # (cx,cy,w,h) → (x1,y1,x2,y2) 좌표 변환
    conf = x[:, 4:5]                 # confidence
    j = x[:, 5:6]                    # class

    mask = conf.flatten() > score_threshold
    filtered = np.concatenate((box, conf, j), axis=1)[mask]

    # confidence 내림차순 정렬
    sorted_indices = np.argsort(-filtered[:, 4])
    x = filtered[sorted_indices]

    print("Shape (box_num, dim):", x.shape)

    if len(x) > 0:
        x1, y1, x2, y2, scores, labels = (
            x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
        )
        keep = []
        idxs = scores.argsort()[::-1]  # 높은 confidence부터 정렬
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = ((x2[i]-x1[i]) * (y2[i]-y1[i]) +
                    (x2[idxs[1:]]-x1[idxs[1:]]) * (y2[idxs[1:]]-y1[idxs[1:]]) - inter)
            iou = inter / (union + 1e-6)

            idxs = idxs[1:][iou <= iou_threshold]
        x = x[keep]
    print("Shape After NMS:", x.shape)


    h, w = frame.shape[:2]
    x1 = np.clip(((x[:, 0] - pad[0]) / ratio[0]).astype(int), 0, w - 1)
    y1 = np.clip(((x[:, 1] - pad[1]) / ratio[1]).astype(int), 0, h - 1)
    x2 = np.clip(((x[:, 2] - pad[0]) / ratio[0]).astype(int), 0, w - 1)
    y2 = np.clip(((x[:, 3] - pad[1]) / ratio[1]).astype(int), 0, h - 1)
    x[:, 0:4] = np.stack([x1, y1, x2, y2], axis=1)

    for i, (x1, y1, x2, y2, score, label) in enumerate(x):
        print(f"Box {i+1}: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}), "
              f"score={score:.2f}, class={int(label)}")


    img_vis = frame.copy()
    for r in x:
        x1, y1, x2, y2, conf, label = r
        label = int(label)
        cls_name = classes[label]
        color = colors[label]

        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img_vis, pt1, pt2, color, 2)

        text = f"{cls_name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_vis, (pt1[0], pt1[1] - th - baseline),
                      (pt1[0] + tw, pt1[1]), color, -1)
        cv2.putText(img_vis, text, (pt1[0], pt1[1] - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # show_img(img_vis, "Final Detection with NMS")
    save_path = f"result.jpg"
    cv2.imwrite(save_path, img_vis)
    print(f"[IMAGE] saved {save_path}")
    


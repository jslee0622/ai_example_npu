import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics.utils import ops
from dx_engine import InferenceEngine

# def show_img(img, title="Image", size=(10,10)): # Img 확인을 위한 함수
#     plt.figure(figsize=size)
#     if img.ndim == 2:
#         plt.imshow(img, cmap="gray")
#     else:
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis("off")
#     plt.show()

model_path = "./assets/models/YOLOv5Pose640_1.dxnn"
img_path = "./assets/inputs/play.jpg"  # 이미지 위치
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
image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) # 모델에 맞게 RGB로 변환

pad = (dw, dh)

#show_img(image_input, "Letter Boxed Image")


ie = InferenceEngine(model_path)

ie_outputs = ie.run([image_input]) # 추론 수행
print("Raw Output Shape:", [o.shape for o in ie_outputs])
print("Raw Output Sample:", ie_outputs[0][0][:5])  # 일부 샘플 출력

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

kpt_count = 17
layers = [
            {
                "name":"0",
                "stride": 8,
                "anchor_width": [
                    19,
                    44,
                    38
                ],
                "anchor_height": [
                    27,
                    40,
                    94
                ]
            },
            {
                "name":"1",
                "stride": 16,
                "anchor_width": [
                    96,
                    86,
                    180
                ],
                "anchor_height": [
                    68,
                    152,
                    137
                ]
            },
            {
                "name":"2",
                "stride": 32,
                "anchor_width": [
                    140,
                    303,
                    238
                ],
                "anchor_height": [
                    301,
                    264,
                    542
                ]
            },
            {
                "name":"3",
                "stride": 64,
                "anchor_width": [
                    436,
                    739,
                    925
                ],
                "anchor_height": [
                    615,
                    380,
                    792
                ]
            }
        ]

ie_output0 = ie_outputs[0][0] # (box 후보 수, feature 크기)
num_det = ie_output0.shape[0] # 검출된 박스 수
decoded_tensor = []

for detected_idx in range(num_det):
        tensor = np.zeros((5 + kpt_count * 3), dtype=float)

        data = ie_output0[detected_idx].tobytes()

        box = np.frombuffer(data[0:16], np.float32)
        gy, gx, anchor, layer = np.frombuffer(data[16:20], np.uint8)
        score = np.frombuffer(data[20:24], np.float32).item()
        label = np.frombuffer(data[24:28], np.uint32).item()
        kpts = np.frombuffer(data[28:232], np.float32)

        if layer >= len(layers):
            break

        w = layers[layer]["anchor_width"][anchor]
        h = layers[layer]["anchor_height"][anchor]
        s = layers[layer]["stride"]

        grid = np.array([gx, gy], np.float32)
        anchor_wh = np.array([w, h], np.float32)

        xc = (grid - 0.5 + (box[0:2] * 2)) * s
        wh = (box[2:4] ** 2) * 4 * anchor_wh
        box_decoded = np.concatenate([xc, wh], axis=0)

        kpts_proc = np.zeros_like(kpts)
        for k in range(kpt_count):
            kx, ky, kc = kpts[k*3:(k+1)*3]
            px = (kx * 2.0 - 0.5 + gx) * s
            py = (ky * 2.0 - 0.5 + gy) * s
            kc = sigmoid(kc)
            kpts_proc[k*3:(k+1)*3] = [px, py, kc]

        tensor[:4] = box_decoded
        tensor[4] = score
        tensor[5:5 + kpt_count * 3] = kpts_proc

        decoded_tensor.append(tensor)

if len(decoded_tensor) == 0:
            decoded_tensor = np.zeros((5 + kpt_count * 3), dtype=float)

decoded_output = np.stack(decoded_tensor)
print("Decoded Tensor Shape:", decoded_output.shape)

x = np.squeeze(decoded_output)
score_threshold = 0.3
iou_threshold = 0.4

if x.ndim <= 1 or len(x) == 0:
    print("No detections")


else:
    box = ops.xywh2xyxy(x[..., :4])  # (cx,cy,w,h) → (x1,y1,x2,y2)
    conf = x[:, 4:5]                 # confidence
    j = x[:, 5:6]                    # class

    kpts = x[..., 5:5 + kpt_count * 3] # keypoints

    mask = conf.flatten() > score_threshold
    filtered = np.concatenate((box, conf, j, kpts), axis=1)[mask]

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
    # for i, (x1, y1, x2, y2, score, label) in enumerate(x):
    #     print(f"Box {i}: ({float(x1)}, {float(y1)}) , ({float(x2)}, {float(y2)}), "
    #           f"score={score:.2f}, class={float(label)}")

    h, w = frame.shape[:2]
    x1 = np.clip(((x[:, 0] - pad[0]) / ratio[0]).astype(int), 0, w - 1)
    y1 = np.clip(((x[:, 1] - pad[1]) / ratio[1]).astype(int), 0, h - 1)
    x2 = np.clip(((x[:, 2] - pad[0]) / ratio[0]).astype(int), 0, w - 1)
    y2 = np.clip(((x[:, 3] - pad[1]) / ratio[1]).astype(int), 0, h - 1)
    x[:, 0:4] = np.stack([x1, y1, x2, y2], axis=1)

    kpts = x[:, 6:].reshape(-1, kpt_count, 3)

    for det_id in range(len(kpts)):
        for k_id, (kx, ky, kc) in enumerate(kpts[det_id]):
            if kc > 0.5:
                px = int((kx - pad[0]) / ratio[0])
                py = int((ky - pad[1]) / ratio[1])
                # clip 해서 이미지 내부로 제한
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                kpts[det_id, k_id, 0] = px
                kpts[det_id, k_id, 1] = py
            else:
                # 신뢰도 낮으면 좌표를 -1 같은 값으로 처리해도 됨
                kpts[det_id, k_id, 0] = -1
                kpts[det_id, k_id, 1] = -1

            x[:, 6:] = kpts.reshape(len(x), -1)

    # for i, (x1, y1, x2, y2, score, label) in enumerate(x):
    #     print(f"Box {i}: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)}), "
    #           f"score={score:.2f}, class={int(label)}")

    img_vis = frame.copy()
    for r in x:
        # --- 바운딩 박스 및 클래스 정보 ---
        x1, y1, x2, y2, conf, label = r[:6]
        label = int(label)

        color = tuple(np.random.randint(0, 256, 3).tolist())

        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img_vis, pt1, pt2, color, 2)

        text = "Person"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_vis, (pt1[0], pt1[1] - th - baseline),
                      (pt1[0] + tw, pt1[1]), color, -1)
        cv2.putText(img_vis, text, (pt1[0], pt1[1] - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # --- Keypoints 시각화 ---
        kpts = r[6:].reshape(kpt_count, 3)

        # 각 keypoint 점 찍기
        for (kx, ky, kc) in kpts:
            if kc > 0.5:
                px = int(kx)
                py = int(ky)
                
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(img_vis, (px, py), 6, (0,0,255), -1)

        # skeleton 연결
        skeleton = [(5,7), (7,9), (6,8), (8,10),
                    (11,13), (13,15), (12,14), (14,16),
                    (5,6), (11,12), (5,11), (6,12)]
        for (p1, p2) in skeleton:
            if kpts[p1][2] > 0.5 and kpts[p2][2] > 0.5:

                p1x, p1y = int(kpts[p1][0]), int(kpts[p1][1])
                p2x, p2y = int(kpts[p2][0]), int(kpts[p2][1])
                # --- IGNORE ---

                cv2.line(img_vis, (p1x, p1y), (p2x, p2y), (255,0,0), 4)



    save_path = f"result.jpg"
    cv2.imwrite(save_path, img_vis)
    print(f"[IMAGE] saved {save_path}")
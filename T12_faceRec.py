import os
import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

##
## 경로 설정
##

##
## Global 변수 및 상수 설정
##
CFG_PATH_MODEL1 = "./assets/models/YOLOV5S_Face-1.dxnn"                 # 얼굴 검출 모델 파일명
CFG_PATH_MODEL2 = "./assets/models/ms1mv2_r50_pfc_dxcom_v1.38.1.dxnn"   # 얼굴 인식 모델 파일명
CFG_PATH_GALLERY = "./gallery"                                          # 갤러리 폴더 경로

CFG_PATH_WEIGHTS_ADD1 = "./assets/weights/add1.npy"
CFG_PATH_WEIGHTS_MUL1 = "./assets/weights/mul1.npy"
CFG_PATH_WEIGHTS_ADD2 = "./assets/weights/add2.npy"

num_anchors = 3
anchors = [
    np.array([[4,5],  [8,10],  [13,16]]),   
    np.array([[146,217],  [231,300],  [335,433]]),
    np.array([[23,29],  [43,55],  [73,105]])  
]
strides = [8, 32, 16]

add1 = np.load(CFG_PATH_WEIGHTS_ADD1)
mul1 = np.load(CFG_PATH_WEIGHTS_MUL1)
add2 = np.load(CFG_PATH_WEIGHTS_ADD2)

io = InferenceOption()

##
##
##
def sigmoid_np(x):
    x = np.clip(x, -88.0, 88.0, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x, dtype=np.float64))

##
##
##
def face_yolo_postprocess_numpy(outputs, anchors, strides):
    batch = outputs[0].shape[0]
    processed = []

    for idx, output in enumerate(outputs):
        h, w = output.shape[1], output.shape[2]
        stride = np.float32(strides[idx])
        anchor = anchors[idx].astype(np.float32)
        num_anchors = anchor.shape[0]

        # (1, H, W, 3, 16)
        output = output.astype(np.float32).reshape(batch, h, w, num_anchors, 16)

        # Sigmoid 적용
        output[..., 0:2] = sigmoid_np(output[..., 0:2])   # cx, cy
        output[..., 2:4] = sigmoid_np(output[..., 2:4])   # w, h
        output[..., 4]   = sigmoid_np(output[..., 4])     # obj conf
        output[..., 15]  = sigmoid_np(output[..., 15])    # face prob (label)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                     np.arange(h, dtype=np.float32))
        grid = np.stack((grid_x, grid_y), axis=-1) 

        #print("grid shape:", grid.shape)
        #print("anchor shape:", anchor.shape)

        xy = ((output[..., 0:2] * 2.0) - 0.5 + grid[None, ..., None, :]) * stride
        wh = ((output[..., 2:4] * 2.0) ** 2) * anchor[None, None, None, :, :]

        lm = output[..., 5:15].reshape(batch, h, w, num_anchors, 5, 2)

        #print("lm shape before:", lm.shape)

        lm = (lm * anchor[None, None, None, :, None, :]) + (grid[None, :, :, None, None, :] * stride)

        # concat [cx, cy, w, h, score, landmarks(10), face_prob]
        box = np.concatenate(
            (
                xy,                            # (2)
                wh,                            # (2)
                output[..., 4:5],                 # (1)
                lm.reshape(batch, h, w, num_anchors, 10),  # (10)
                output[..., 15:16],            # (1)
            ),
            axis=-1,
        )  # total = 16 channels again

        # (1, H*W*3, 16)
        box = box.reshape(batch, -1, 16).astype(np.float32)
        processed.append(box)

    # (1, total, 16)
    return np.concatenate(processed, axis=1).astype(np.float32)

##
##
##
def feature_postprocess_numpy(features):
    """
    features: np.ndarray, shape [1, 512]
    순서: add1 → mul1 → add2
    """

    features = (features + add1) * mul1 + add2
    
    return features.astype(np.float32)
    
##
##
##
def preprocess(image_src, new_shape=(640, 640), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB):
    # 이미지 크기를 모델 입력 크기에 맞추되, 원본 비율을 유지하며 padding을 추가하는 전처리 함수
    # image_src: 입력 이미지 (numpy array, BGR)
    # new_shape: 원하는 출력 크기 (기본 640x640)
    # fill_color: padding 시 채울 색상 (기본 [114,114,114] = 회색)
    # format: 색상 공간 변환 코드

    # 원본 이미지의 높이, 너비
    src_shape = image_src.shape[:2]  # (height, width)

    # new_shape이 int라면 pair로 변환
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 축소/확대 비율 계산 (세로와 가로 중 더 작은 비율 선택 → 종횡비 유지)
    r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])
    ratio = r, r

    # padding 없이 리사이즈했을 때의 크기
    new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))

    # 남는 공간(padding 크기) 계산
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    # padding은 좌우/상하 균등하게 분배
    dw /= 2
    dh /= 2

    # 종횡비가 다르면 resize 실행
    if src_shape[::-1] != new_unpad:
        image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 실제 padding 적용 (top, bottom, left, right)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_new = cv2.copyMakeBorder(
        image_src, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=fill_color
    )

    # 색상 변환
    if format is not None:
        image_new = cv2.cvtColor(image_new, format)

    # 전처리된 이미지, resize 비율, padding 크기 반환
    return image_new, ratio, (dw, dh)

##
##
##
def decode(ie_output_dict, score_threshold=0.25, kpt_count=5):
    # ie_output_dict: 추론 엔진에서 나온 결과
    # score_threshold: 얼굴을 검출할 최소 confidence 값
    # kpt_count: 사용할 keypoint 개수 (기본 5개: 눈, 코, 입꼬리)

    # 모델 출력에서 첫 번째 값 꺼내기
    ie_output = ie_output_dict[0]

    decoded_tensor = []
    for det in ie_output:
        # det = [cx, cy, w, h, score, kpt1_x, kpt1_y, ..., kpt5_x, kpt5_y, label]
        score = float(det[4])  # confidence 값
        if score < score_threshold:
            # confidence가 threshold보다 낮으면 무시
            continue

        box = det[0:4]                   # 바운딩 박스 정보 (cx, cy, w, h)
        kpts = det[5:5 + kpt_count*2]    # keypoints 좌표 (x, y) * kpt_count
        label = int(det[-1])             # 클래스 레이블 (마지막 값)

        # [box(4), score(1), keypoints(2*kpt_count)] 형태의 텐서 생성
        tensor = np.zeros((5 + kpt_count*2), dtype=float)
        tensor[:4] = box                 # 바운딩 박스 저장
        tensor[4] = score                # score 저장
        tensor[5:5+kpt_count*2] = kpts   # keypoints 저장
        decoded_tensor.append(tensor)    # 결과 리스트에 추가

    if len(decoded_tensor) == 0:
        # 얼굴이 검출되지 않았다면 0으로 채워진 벡터 반환
        return np.zeros((5 + kpt_count*2), dtype=float)

    # 리스트를 numpy 배열로 변환
    return np.stack(decoded_tensor)

##
##
##
def nms(dets, iou_thr=0.5):
    """
    Args:
        dets: ndarray [N, 6]
              (x1, y1, x2, y2, score, label)
              - (x1, y1): 좌상단 좌표
              - (x2, y2): 우하단 좌표
              - score: confidence 점수
              - label: 클래스 라벨
        iou_thr: IoU threshold (겹침 정도 기준, 기본값 0.5)

    Returns:
        keep: 선택된 박스들의 index 리스트
    """

    if len(dets) == 0:
        return dets  # 박스가 없으면 그대로 반환

    # 좌표와 점수, 레이블 분리
    x1, y1, x2, y2, scores, labels = (
        dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4], dets[:, 5]
    )

    keep = []  # 최종 선택된 박스 인덱스 저장
    # 점수(score)가 큰 순서대로 정렬된 인덱스
    idxs = scores.argsort()[::-1]

    while len(idxs) > 0:
        # 현재 가장 점수가 높은 박스 선택
        i = idxs[0]
        keep.append(i)

        # 선택된 박스(i)와 나머지 박스들 간의 좌표 비교
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        # 겹치는 영역(intersection) 계산
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # union = (각 영역 넓이 합 - 교집합 넓이)
        union = ((x2[i]-x1[i]) * (y2[i]-y1[i]) +
                 (x2[idxs[1:]]-x1[idxs[1:]]) * (y2[idxs[1:]]-y1[idxs[1:]]) - inter)

        # IoU(Intersection over Union) 계산
        iou = inter / (union + 1e-6)

        # IoU가 threshold 이하인 것만 남김 (겹침이 심한 박스는 제거)
        idxs = idxs[1:][iou <= iou_thr]

    return keep

##
##
##
def warp_face(img, landmarks):
    # 얼굴 정렬 함수 (Face Alignment)
    # 입력된 얼굴 랜드마크를 기준 좌표에 맞게 변환하여
    # 동일한 크기(112x112)로 정렬된 얼굴 이미지를 생성한다.

    # src_points: 입력 이미지에서 검출된 랜드마크 좌표 (5개 점: 눈, 코, 입 양쪽)
    src_points = np.array(landmarks, dtype=np.float32)  # shape = (5, 2)

    # dst_points: 정렬 기준이 되는 표준 얼굴 랜드마크 좌표
    # (112x112 크기에서의 기준 위치, 보통 학습 데이터셋에서 정의)
    dst_points = np.array([
        [38.2946, 51.6963],   # 왼쪽 눈
        [73.5318, 51.5014],   # 오른쪽 눈
        [56.0252, 71.7366],   # 코
        [41.5493, 92.3655],   # 왼쪽 입
        [70.7299, 92.2041],   # 오른쪽 입
    ], dtype=np.float32)

    # 입력 랜드마크(src_points) → 기준 랜드마크(dst_points)로 맞추는
    # Affine 변환 행렬 M 계산 (scale, rotation, translation 포함)
    M, _ = cv2.estimateAffinePartial2D(src_points, dst_points)

    # img를 변환 행렬 M으로 warp → 112x112 크기로 맞춘 얼굴 이미지 생성
    aligned = cv2.warpAffine(img, M, (112, 112))

    return aligned  # 정렬된 얼굴 이미지 반환

##
##
##
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)

##
##
##
def extract_feature(ie, ie_recog, frame):
    """
    얼굴 특징(embedding) 벡터를 추출하는 함수
    1) 얼굴 검출 → 2) NMS → 3) 랜드마크 기반 정렬 → 4) 얼굴 임베딩 추출
    """

    score_threshold = 0.25     # 얼굴 검출 최소 confidence
    iou_threshold = 0.4        # NMS에서 사용할 IoU 임계값
    kpt_count = 5              # 사용할 keypoint 개수 (왼쪽 눈, 오른쪽 눈, 코, 입 양쪽)

    # (1) 얼굴 검출 모델 입력 크기 계산
    #(ie.get_input_size() = 모델 입력 크기(H*W*C)
    input_size = int(np.sqrt(ie.get_input_size() / 3))

    # (2) 입력 이미지를 모델 입력 크기에 맞게 전처리
    # - 비율 유지 resize + padding + 색상공간 변환(BGR → RGB)
    image_input, ratio, pad = preprocess(
        frame, new_shape=(input_size, input_size),
        fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB
    )

    # use_ort == False 일 경우, NPU에 Input으로 바로 넣기 위해 Reshape이 필요하나, 메모리 상 동일하므로 생략 가능
    # if(io.get_use_ort()==False):
    #     image_input = image_input.astype(np.uint8).reshape(1, input_size, input_size*3)

    # (3) 얼굴 검출 모델 실행
    ie_output = (ie.run([image_input]))
    
    if(io.get_use_ort()==False):  # ONNX Runtime 사용 시 후처리 필요
        #print("Using DX Engine")
        ie_output1 = face_yolo_postprocess_numpy(
            ie_output, anchors=anchors, strides=strides
        )
    else:
        #print("Using ONNX Runtime")
        ie_output1 = ie_output[0]
    
    # (4) 모델 출력 후처리 → 바운딩박스, score, keypoints 추출
    decoded_tensor = decode(
        ie_output1, score_threshold=score_threshold, kpt_count=kpt_count
    )

    # (5) 차원 축소 (불필요한 차원 제거)
    x = np.squeeze(decoded_tensor)
    if x.ndim == 1:
        # 검출된 얼굴이 없을 경우
        return None

    # (6) confidence(score) 기준 필터링
    x = x[x[..., 4] > score_threshold]
    if x.shape[0] == 0:
        return None

    # (7) 바운딩 박스와 keypoints 정리
    all_dets = []
    for det in x:
        cx, cy, w, h, conf = det[:5]  # 중심좌표(cx, cy), 폭, 높이, confidence
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        kpts = det[5:5 + kpt_count*2].reshape(kpt_count, 2)  # keypoints 좌표 (5개 점)
        # [x1, y1, x2, y2, score, label, keypoints...] 형태로 저장
        all_dets.append([x1, y1, x2, y2, conf, 0, *kpts.flatten()])
    all_dets = np.array(all_dets)

    # (8) NMS 적용 → 겹치는 박스 제거, score 높은 것만 남김
    keep = nms(all_dets[:, :6], iou_thr=iou_threshold)
    final_dets = all_dets[keep]

    # (9) 원본 이미지 크기 (좌표 보정에 필요)
    h, w = frame.shape[:2]
    results = []  # 최종 얼굴 특징과 bbox를 담을 리스트

    # (10) 검출된 얼굴 후보들 반복 처리
    for det in final_dets:
        kpts = det[6:].reshape(kpt_count, 2)  # (5,2) keypoints 좌표

        # (11) padding, resize 비율을 보정하여 원본 이미지 좌표로 변환
        landmarks = []
        for (kx, ky) in kpts:
            px = int((kx - pad[0]) / ratio[0])
            py = int((ky - pad[1]) / ratio[1])
            # 이미지 경계 밖으로 나가지 않도록 제한
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            landmarks.append((px, py))

        # (12) 랜드마크를 기준으로 얼굴 정렬 (112x112 크기)
        aligned_face = warp_face(frame, landmarks)

        # (13) 얼굴 인식 모델 입력 크기에 맞게 reshape (배치 차원 포함)
        fr_input = aligned_face.astype(np.uint8).reshape(1, 112, 112, 3)

        if(io.get_use_ort()==False):
            
            #resshape 1x112x336
            fr_input = fr_input.astype(np.uint8).reshape(1, 112, 336)

            zeros = np.zeros((1, 112, 48), dtype=fr_input.dtype)

            fr_input = np.concatenate((fr_input, zeros), axis=2)

        # (14) 얼굴 인식 모델 실행 → 임베딩 벡터(128D, 512D 등) 추출
        fr_outputs = ie_recog.run([fr_input])

        if(io.get_use_ort()==False):
            fr_outputs = np.squeeze(fr_outputs[0]).reshape(1, 512)
            # print("here")

            feat = feature_postprocess_numpy(fr_outputs)
            feat = np.array(feat, copy=False).reshape(-1)
            feat = feat / np.linalg.norm(feat)
        else:
            # print("Using ONNX Runtime for feature extraction")
            # print("fr_outputs[0] shape:", fr_outputs[0].shape)
            feat = np.array(fr_outputs[0], copy=False).reshape(-1)
            feat = feat / np.linalg.norm(feat)
        # print("feat example:", feat, "... (len=", len(feat), ")")

        # (15) 특징 벡터와 바운딩 박스를 함께 저장
        x1, y1, x2, y2 = det[:4]
        # 좌표를 원본 이미지에 맞게 보정
        x1 = int(((x1) - pad[0]) / ratio[0])
        y1 = int(((y1) - pad[1]) / ratio[1])
        x2 = int(((x2) - pad[0]) / ratio[0])
        y2 = int(((y2) - pad[1]) / ratio[1])
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        results.append({"feat": feat, "bbox": (x1, y1, x2, y2)})

    # (16) 모든 얼굴 특징 + bbox 리스트 반환 (없으면 None 반환)
    return results if len(results) > 0 else None

##
##
##
def build_gallery(ie_align, ie_recog, image_paths):
    """
    갤러리(등록된 얼굴 데이터베이스)를 구축하는 함수
    - 주어진 폴더 내 이미지에서 얼굴 특징 벡터 추출
    - 각 얼굴에 랜덤 색상 지정
    - 반환 형식: {이름: {"feat": 특징벡터, "color": (B,G,R)}}
    """

    gallery = {}  # 최종적으로 반환할 딕셔너리

    # (1) 폴더 내 파일들 순회
    for fname in os.listdir(image_paths):
        fpath = os.path.join(image_paths, fname)
        img = cv2.imread(fpath)  # 이미지 읽기
        if img is None:
            # 이미지 파일을 읽을 수 없는 경우
            print(f"[WARNING] Could not read image {fpath}, skipping.")
            continue

        # (2) 얼굴 특징 추출
        # extract_feature는 리스트를 반환하며, 각 원소는 {"feat": ..., "bbox": ...} 구조
        results = extract_feature(ie_align, ie_recog, img)

        if results is not None and len(results) > 0:
            # 파일명에서 확장자 제거 → 이름으로 사용
            name, _ = os.path.splitext(fname)
            # 첫 번째 얼굴의 feature만 사용 (한 장에 얼굴 하나라고 가정)
            feat = results[0]["feat"]

            # print(f"[INFO] Feature example for {name}: {feat[:5]}... (len={len(feat)})")
            # (3) 랜덤 색상 생성 (0~255 범위에서 BGR 값 선택)
            color = tuple(np.random.randint(0, 256, size=3).tolist())

            # (4) gallery 딕셔너리에 저장
            # - key: 이름
            # - value: {"feat": 얼굴 임베딩 벡터, "color": 랜덤 색상}
            gallery[name] = {"feat": feat, "color": color}
            print(f"[INFO] Added {name} to gallery with random color {color}.")
        else:
            # 얼굴 검출 실패한 경우
            print(f"[WARNING] No face detected in {fpath}, skipping.")

    # (5) 모든 이미지 처리 후 반환
    return gallery

##
##
##
def run_inference_face(ie, ie_recog, frame, gallery,
                       kpt_count=5):
    """
    얼굴 인식 실행 함수
    - 입력 프레임에서 얼굴 검출 및 특징 벡터 추출 (extract_feature 호출)
    - 갤러리에 등록된 얼굴들과 cosine similarity로 비교
    - 가장 유사한 이름/스코어를 bbox 위에 표시
    """

    img_vis = frame.copy()

    # (1) 프레임에서 얼굴 특징 + bbox 추출
    results = extract_feature(ie, ie_recog, frame)
    if results is None or len(results) == 0:
        # 얼굴이 검출되지 않은 경우 → 원본 그대로 반환
        return img_vis

    # (2) 검출된 모든 얼굴에 대해 갤러리와 비교
    for res in results:
        feat = res["feat"]            # 얼굴 임베딩 벡터
        (x1, y1, x2, y2) = res["bbox"]  # 얼굴 바운딩 박스 좌표

        # (3) 갤러리와 cosine similarity 비교 → 가장 높은 점수 찾기
        best_name, best_score = "Unknown", 0.0
        for name, g_data in gallery.items():
            g_feat = g_data["feat"]
            sim = cosine_similarity(feat, g_feat)
            # print(name, sim)
            if sim > best_score:
                best_name, best_score = name, sim

        # (4) threshold(0.5) 이상이면 이름 + score 표시, 아니면 Unknown
        label = f"{best_name}:{best_score:.2f}" if best_score > 0.5 else "Unknown"

        # 색상 불러오기 (Unknown은 흰색)
        color = gallery[best_name]["color"] if best_name in gallery else (255, 255, 255)

        # (5) bbox와 label을 프레임에 그리기
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # (6) 결과 이미지 반환
    return img_vis

##
##
##
def run():
    """
    얼굴 검출 + 인식 실행 함수 (비디오 입력 전용)
    - 웹캠을 통해 실시간 얼굴 인식 수행
    - 'c' 키: 얼굴 캡처 후 갤러리에 저장
    - 'd' 키: 갤러리 이미지 삭제
    - 'q' 키: 종료
    """
    
    # (1) 모델 경로 설정
    model_path1 = CFG_PATH_MODEL1     # 얼굴 검출 모델
    model_path2 = CFG_PATH_MODEL2  # 얼굴 인식 모델
    
    # io.set_use_ort(False)  # DX Engine 사용 설정
    # print("Get bound options:", io.get_bound_option())

    # (2) 추론 엔진 초기화
    ie = InferenceEngine(model_path1, io)
    ie_recog = InferenceEngine(model_path2, io)
    print("[INFO] Inference engines initialized.")
    # (3) 갤러리 로드 (등록된 얼굴 DB)
    gallery = build_gallery(ie, ie_recog, CFG_PATH_GALLERY)

    # (4) 웹캠 열기
    cap = cv2.VideoCapture(0)  # 기본 카메라 장치
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        exit(1)

    print("\n=== Face Recognition... ===")
    print("Controls on Webcam Window:")
    print("  'c' key: Capture face and save to gallery")
    print("  'd' key: Delete image from gallery")
    print("  'q' key: Quit")
    print("================================\n")

    # (5) 루프 시작
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 실시간 얼굴 인식 실행
        img_vis = run_inference_face(ie, ie_recog, frame, gallery)
        cv2.imshow("result", img_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 종료
            break

        # 'c' 키: 얼굴 캡처 후 갤러리에 저장
        elif key == ord('c'):
            # (프레임에서 얼굴 특징 추출
            results = extract_feature(ie, ie_recog, frame)

            if results is None or len(results) == 0:
                print("[CAPTURE] No face detected in frame")
                continue

            # (검출된 얼굴들 순회
            i = 1
            h, w = frame.shape[:2]
            for res in results:
                feat = res["feat"]      # 얼굴 feature (갤러리에는 이미지 저장만)
                (x1, y1, x2, y2) = res["bbox"]

                # bbox 확장 (머리/턱 여유 공간 포함)
                bw = (x2 - x1) / 2 * 0.5   # bbox 폭의 절반 * 비율
                bh = (y2 - y1) / 2 * 0.5   # bbox 높이의 절반 * 비율
                x1 = max(0, int(x1 - bw))
                y1 = max(0, int(y1 - bh))
                x2 = min(w, int(x2 + bw))
                y2 = min(h, int(y2 + bh))
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size > 0:
                    win_name = f"Face {i}"
                    cv2.imshow(win_name, face_crop)
                    cv2.waitKey(1)

                    # 사용자에게 이름 입력받음
                    name = input(f"Enter name for face {i}: ").strip()
                    if not name:
                        name = f"face_{i}"
                    i += 1

                    # crop 얼굴 저장
                    save_path = os.path.join(CFG_PATH_GALLERY, f"{name}.jpg")
                    cv2.imwrite(save_path, face_crop)
                    print(f"[CAPTURE] Saved cropped face to {save_path}")

                    cv2.destroyWindow(win_name)
                else:
                    print(f"[CAPTURE] No valid face crop found for detection #{i}")

            # (5) 갤러리 갱신
            gallery = build_gallery(ie, ie_recog, CFG_PATH_GALLERY)

        # 'd' 키: 갤러리에서 이미지 삭제
        elif key == ord('d'):
            gallery_files = [f for f in os.listdir(CFG_PATH_GALLERY) if f.lower().endswith((".jpg", ".png"))]
            if not gallery_files:
                print("[DELETE] Gallery is empty.")
            else:
                print("\n[DELETE] Gallery Images:")
                for idx, fname in enumerate(gallery_files, start=1):
                    print(f"{idx}: {fname}")
                try:
                    sel = int(input("Enter number to delete: ").strip())
                    if 1 <= sel <= len(gallery_files):
                        del_path = os.path.join(CFG_PATH_GALLERY, gallery_files[sel-1])
                        os.remove(del_path)
                        print(f"[DELETE] Removed {del_path}")
                        gallery = build_gallery(ie, ie_recog, CFG_PATH_GALLERY)
                    else:
                        print("[DELETE] Invalid number.")
                except ValueError:
                    print("[DELETE] Invalid input.")

    cap.release()
    cv2.destroyAllWindows()

##
##
##
if __name__ == "__main__":
    run()
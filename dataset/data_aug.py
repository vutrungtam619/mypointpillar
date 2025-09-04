import numpy as np
import cv2
from config.config import config

def random_flip_fusion(data_dict, flip_prob=0.7):
    """
    Flip đồng bộ cả ảnh (numpy RGB) và point cloud theo trục Y.
    Trả về image dạng numpy (H,W,3), để model tự chuẩn hóa.
    """
    if np.random.rand() < flip_prob:
        # --- Flip LiDAR ---
        pts, gt_bboxes_3d = data_dict['pts'], data_dict['gt_bboxes_3d']
        pts[:, 1] = -pts[:, 1]
        gt_bboxes_3d[:, 1] = -gt_bboxes_3d[:, 1]
        gt_bboxes_3d[:, 6] = -gt_bboxes_3d[:, 6] + np.pi
        data_dict['pts'] = pts
        data_dict['gt_bboxes_3d'] = gt_bboxes_3d

        # --- Flip image (numpy RGB) ---
        image = data_dict['image']
        data_dict['image'] = np.ascontiguousarray(cv2.flip(image, 1))

        # --- Update calibration ---
        calib = data_dict['calib_info']
        P2 = calib['P2'].copy()
        w = image.shape[1]
        P2[0, 2] = w - P2[0, 2]  # cx -> W - cx
        P2[0, 3] = -P2[0, 3]     # Tx -> -Tx
        calib['P2'] = P2
        data_dict['calib_info'] = calib

    return data_dict

def color_jitter_fusion(data_dict, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, prob=0.7):
    """
    Augment màu sắc cho ảnh trong data_dict (RGB numpy HxWx3).
    Tự random trong hàm, chỉ thay đổi ảnh, pts/bbox/calib giữ nguyên.
    """
    if np.random.rand() > prob:
        return data_dict  # không áp dụng

    image = data_dict['image'].astype(np.float32) / 255.0

    # Brightness
    if brightness > 0:
        factor = 1.0 + np.random.uniform(-brightness, brightness)
        image *= factor

    # Contrast
    if contrast > 0:
        mean = image.mean(axis=(0,1), keepdims=True)
        factor = 1.0 + np.random.uniform(-contrast, contrast)
        image = (image - mean) * factor + mean

    # Saturation
    if saturation > 0:
        gray = image.mean(axis=2, keepdims=True)
        factor = 1.0 + np.random.uniform(-saturation, saturation)
        image = (image - gray) * factor + gray

    # Hue shift
    if hue > 0:
        hsv = cv2.cvtColor((image*255).clip(0,255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + np.random.uniform(-hue*180, hue*180)) % 180
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    data_dict['image'] = np.clip(image * 255, 0, 255).astype(np.uint8)
    return data_dict



def point_range_filter(data_dict, point_range=config['pc_range']):
    pts = data_dict['pts']
    mask = (
        (pts[:, 0] > point_range[0]) &
        (pts[:, 1] > point_range[1]) &
        (pts[:, 2] > point_range[2]) &
        (pts[:, 0] < point_range[3]) &
        (pts[:, 1] < point_range[4]) &
        (pts[:, 2] < point_range[5])
    )
    data_dict['pts'] = pts[mask]
    return data_dict


def object_range_filter(data_dict, object_range=config['pc_range']):
    """Lọc ground truth bbox 3D theo vị trí."""
    gt_bboxes_3d, gt_labels = data_dict['gt_bboxes_3d'], data_dict['gt_labels']
    gt_names, difficulty = data_dict['gt_names'], data_dict['difficulty']

    mask = (
        (gt_bboxes_3d[:, 0] > object_range[0]) &
        (gt_bboxes_3d[:, 1] > object_range[1]) &
        (gt_bboxes_3d[:, 0] < object_range[3]) &
        (gt_bboxes_3d[:, 1] < object_range[4])
    )

    data_dict['gt_bboxes_3d'] = gt_bboxes_3d[mask]
    data_dict['gt_labels'] = gt_labels[mask]
    data_dict['gt_names'] = gt_names[mask]
    data_dict['difficulty'] = difficulty[mask]
    
    return data_dict


def points_shuffle(data_dict):
    pts = data_dict['pts']
    idx = np.arange(len(pts))
    np.random.shuffle(idx)
    data_dict['pts'] = pts[idx]
    return data_dict


def fusion_data_augment(data_dict):
    # 1. Random flip đồng bộ
    data_dict = random_flip_fusion(data_dict)

    # 2. Color jitter cho ảnh
    data_dict = color_jitter_fusion(data_dict)

    # 3. Point range filter
    data_dict = point_range_filter(data_dict)

    # 4. Object range filter
    data_dict = object_range_filter(data_dict)

    # 5. Point shuffle
    data_dict = points_shuffle(data_dict)

    return data_dict

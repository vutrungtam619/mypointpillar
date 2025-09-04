import cv2
import torch
import numpy as np
from dataset.kitti import Kitti
from utils.process import project_point_to_camera, bbox3d2corners_camera, points_camera2image, keep_bbox_from_image_range

def draw_projected_box3d(image, corners, color=(0, 0, 255), thickness=1):
    """
    Vẽ box 3D đã được project lên ảnh (12 cạnh).
    Args:
        image: numpy array (H,W,3)
        corners: numpy array (8,2) chứa pixel tọa độ các đỉnh
        color: màu (BGR)
        thickness: độ dày
    """
    corners = corners.astype(int)

    # Kiểm tra xem có corner nào nằm ngoài ảnh quá nhiều không
    h, w = image.shape[:2]
    if np.all((corners[:, 0] < 0) | (corners[:, 0] >= w)) or \
       np.all((corners[:, 1] < 0) | (corners[:, 1] >= h)):
        return image

    # 12 cạnh của cuboid
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # đáy
        (4, 5), (5, 6), (6, 7), (7, 4),  # nóc
        (0, 4), (1, 5), (2, 6), (3, 7)   # cột dọc
    ]

    for i, j in edges:
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[j])
        cv2.line(image, pt1, pt2, color, thickness)

    return image


if __name__ == "__main__":
    # ======================
    # Chọn id muốn test
    # ======================
    target_id = 113   # ví dụ ảnh 000123.png

    # Khởi tạo dataset (augment được áp dụng khi split="train")
    dataset = Kitti(data_root="kitti", split="train")

    # Lấy index của id
    idx = dataset.sorted_ids.index(target_id)

    # Lấy sample sau augment
    sample = dataset[idx]

    # Tách dữ liệu
    pts = torch.from_numpy(sample["pts"][:, :3])  # (N,3)
    calib = sample["calib_info"]
    image = sample["image"].copy()
    gt_bboxes_3d = sample["gt_bboxes_3d"]  # (M,7) LiDAR coords

    # ======================
    # Project LiDAR points (chấm xanh)
    # ======================
    u, v = project_point_to_camera(pts, calib)
    u = u.cpu().numpy().astype(int)
    v = v.cpu().numpy().astype(int)

    for x, y in zip(u, v):
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # ======================
    # Vẽ GT 3D bboxes (khung đỏ)
    # ======================

    if len(gt_bboxes_3d) > 0:
        # 0. Tạo dict để dùng hàm lọc
        result = {
            "lidar_bboxes": gt_bboxes_3d,
            "labels": sample["gt_names"],
            "scores": np.ones(len(gt_bboxes_3d)),  # dummy score
        }

        # 1. Lọc box nằm trong FOV ảnh
        result = keep_bbox_from_image_range(
            result,
            calib["Tr_velo_to_cam"],
            calib["R0_rect"],
            calib["P2"],
            image.shape[:2]
        )

        # 2. Lấy lại box camera đã được lọc
        bboxes_camera = result["camera_bboxes"]

        if len(bboxes_camera) > 0:
            # 3. Lấy corners 3D trong camera
            corners_cam = bbox3d2corners_camera(bboxes_camera)

            # Giữ lại box nếu >= 50% corners z > 0
            valid_mask = np.mean(corners_cam[:, :, 2] > 0, axis=1) > 0.5
            corners_cam = corners_cam[valid_mask]

            # 4. Project ra ảnh (N,8,2)
            corners_img = points_camera2image(corners_cam, calib["P2"])

            # 5. Vẽ từng box
            for box in corners_img:
                image = draw_projected_box3d(image, box, color=(0, 0, 255), thickness=1)

    # ======================
    # Hiển thị / lưu
    # ======================
    cv2.imshow(f"GT Projection ID {target_id}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f"debug_gt_projection_id{target_id}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
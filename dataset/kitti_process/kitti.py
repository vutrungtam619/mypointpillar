import numpy as np
import os
from config.config import config
from torch.utils.data import Dataset
from utils.io import read_pickle, read_points
from utils.process import bbox_camera2lidar

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def point_range_filter(data_dict, point_range):
    '''
    Args: dict with the following 
        pts [np.ndarray float32, (n, 4)]: total LiDAR points in this item
        gt_bboxes_3d [np.ndarray float32, (m, 7)]: bounding box in LiDAR coordinate
        gt_labels [np.ndarray int32, (m, )]: numerical labels for each object
        gt_names [np.ndarray string, (m, )]: object class name
        num_points_in_gt [np.ndarrat int32, (n, )]: number of points in 1 gt bounding box
        difficulty [np.ndarray float32, (m, )]: 0 is easy, 1 is moderate, 2 is hard, -1 is not classify
        image_shape [tuple int32, (2, )]: image shape in (height, width)
        image_path [string]: full image path
        calib_info [dict]: calib information        
        point_range [list float32, (6)]: [x1, y1, z1, x2, y2, z2]
    Returns: 
        data_dict
    '''
    pts = data_dict['pts']
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    data_dict.update({'pts': pts})
    return data_dict 

class Kitti(Dataset): 
    def __init__(self, data_root, split):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.data_infos = read_pickle(os.path.join(project_root, 'dataset', f'kitti_infos_{split}.pkl'))
        self.sorted_ids = list(self.data_infos.keys())
        self.CLASS = config['CLASSES']
        
    def remove_dontcare(self, annos_info):
        keep_ids = [i for i, name in enumerate(annos_info['name']) if name != 'DontCare']
        for k, v in annos_info.items():
            annos_info[k] = v[keep_ids]
        return annos_info   
    
    def __getitem__(self, index):
        """ Get the information of one item
        Args:
            index [int]: index of item

        Returns: dict with the following, m is the number of objects in item
            pts [np.ndarray float32, (n, 4)]: total LiDAR points in this item
            gt_bboxes_3d [np.ndarray float32, (m, 7)]: bounding box in LiDAR coordinate
            gt_labels [np.ndarray int32, (m, )]: numerical labels for each object
            gt_names [np.ndarray string, (m, )]: object class name
            num_points_in_gt [np.ndarrat int32, (n, )]: number of points in 1 gt bounding box
            difficulty [np.ndarray float32, (m, )]: 0 is easy, 1 is moderate, 2 is hard, -1 is not classify
            image_shape [tuple int32, (2, )]: image shape in (height, width)
            image_path [string]: full image path
            calib_info [dict]: calib information
        """
        data_info = self.data_infos[self.sorted_ids[index]]
        image_info, calib_info, annos_info, velodyne_path = data_info['image'], data_info['calib'], data_info['annos'], data_info['velodyne_path']
        
        pts = read_points(velodyne_path) # np.float32
        
        Tr_velo_to_cam = calib_info['Tr_velo_to_cam'].astype(np.float32)
        R0_rect = calib_info['R0_rect'].astype(np.float32)
        
        annos_info = self.remove_dontcare(annos_info)
        annos_name = annos_info['name']
        annos_location = annos_info['location']
        annos_dimension = annos_info['dimensions']
        rotation_y = annos_info['rotation_y']
        gt_bboxes = np.concatenate([annos_location, annos_dimension, rotation_y[:, None]], axis=1).astype(np.float32)
        gt_bboxes_3d = bbox_camera2lidar(gt_bboxes, Tr_velo_to_cam, R0_rect)
        gt_labels = [self.CLASS.get(name, -1) for name in annos_name]
        
        image_shape = image_info['image_shape']
        image_path = os.path.join(self.data_root, image_info['image_path'])
        
        data_dict = {
            'pts': pts,
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels': np.array(gt_labels),
            'gt_names': annos_name,
            'num_points_in_gt': annos_info['num_points_in_gt'],
            'difficulty': annos_info['difficulty'],
            'image_shape': image_shape,
            'image_path': image_path,
            'calib_info': calib_info
        }
        
        data_dict = point_range_filter(data_dict=data_dict, point_range=config['pc_range'])
        
        return data_dict
    
    def __len__(self):
        return len(self.data_infos)
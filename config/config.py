import os
project_root = os.path.dirname(os.path.dirname(__file__))

config = {
    'CLASSES': {'Pedestrian': 0, 'Cyclist': 1, 'Car': 2},
    'data_root': 'kitti',
    'num_classes': 3,
    'pc_range': [0, -39.68, -3, 69.12, 39.68, 1],
    'voxel_size': [0.16, 0.16, 4],
    'max_voxels': (12000, 40000),
    'max_points': 32,
    'new_shape': (384, 1280),
    'mean': [0.36783523, 0.38706144, 0.3754649],
    'std': [0.31566228, 0.31997792, 0.32575161],
    'ANCHOR' : {
        'ranges': [[0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -0.6, 69.12, 39.68, -0.6], [0, -39.68, -1.78, 69.12, 39.68, -1.78]],
        'sizes': [[0.6, 0.8, 1.73], [0.6, 1.76, 1.73], [1.6, 3.9, 1.56]],
        'rotations': [0, 1.57],
    },
    'checkpoint_dir': os.path.join(project_root, 'checkpoints'),
    'batch_size': 8,
    'num_workers': 4,
    'init_lr': 0.0001,
    'epoch': 30,
    'ckpt_freq': 2,
    'log_freq': 50,
    'log_dir': os.path.join(project_root, 'log')
}
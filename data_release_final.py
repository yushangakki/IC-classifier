import os
import sys
import h5py
import numpy as np
from numpy import array, all
from torch.utils.data import Dataset

# Configure numpy print options
np.set_printoptions(threshold=sys.maxsize)

# Constants
TRAIN_DIR = os.path.dirname("YOUR DIRECTORY")
TEST_DIR = os.path.dirname("YOUR DIRECTORY")

# Mechanical components categories
MECHANICAL_COMPONENTS = [
    'Articulations, eyelets and other articulated joints',
    'Bearing accessories',
    'Bushes',
    'Cap nuts',
    'Castle nuts',
    'Castor',
    'Chain drives',
    'Clamps',
    'Collars',
    'Conventional rivets',
    'Convex washer',
    'Cylindrical pins',
    'Elbow fitting',
    'Eye screws',
    'Fan',
    'Flanged block bearing',
    'Flanged plain bearings',
    'Flange nut',
    'Grooved pins',
    'Helical geared motors',
    'Hexagonal nuts',
    'Hinge',
    'Hook',
    'Impeller',
    'Keys and keyways, splines',
    'Knob',
    'Lever',
    'Locating pins'
    'Locknuts',
    'Lockwashers',
    'Nozzle',
    'Plain guidings',
    'Plates, circulate plates',
    'Plugs',
    'Pulleys',
    'Radial contact ball bearings',
    'Right angular gearings',
    'Right spur gears',
    'Rivet nut',
    'Roll pins',
    'Screws and bolts with countersunk head',
    'Screws and bolts with cylindrical head',
    'Screws and bolts with hexagonal head',
    'Setscrew',
    'Slotted nuts',
    'Snap rings',
    'Socket',
    'Spacers',
    'Split pins',
    'Springs',
    'Spring washers',
    'Square',
    'Square nuts',
    'Standard fitting',
    'Studs',
    'Switch',
    'Taper pins',
    'Tapping screws',
    'Threaded rods',
    'Thrust washers',
    'T-nut',
    'Toothed',
    'T-shape fitting',
    'Turbine',
    'Valve',
    'Washer bolt',
    'Wheel',
    'Wingnuts',
]

def load_data(path):
    """
    Load model and label data from H5 files.
    
    Args:
        path (str): Either 'test' or 'train' to specify which dataset to load
        
    Returns:
        tuple: (model data array, label array)
    """
    h5_path = '/home/leo/label_test-2.h5' if path == 'test' else '/home/leo/label_train-2.h5'
    
    with h5py.File(h5_path, 'r') as hf:
        model = hf['model_test-2'][:].astype('float32')
        label = hf['label_test-2'][:].astype('int64')
        hf.close()
    
    label = label.reshape(len(label), -1)
    return model, label

def gen_data(path):
    """
    Generate dataset from point cloud files and save to H5 format.
    
    Args:
        path (str): Either 'test' or 'train' to specify which dataset to generate
        
    Returns:
        tuple: (0, 0) placeholder return values
    """
    base_dir = TEST_DIR if path == 'test' else TRAIN_DIR
    labels = np.array([[]])
    label_test = []
    
    for i, _ in enumerate(MECHANICAL_COMPONENTS):
        label_now = i
        component_path = os.path.join(base_dir, str(label_now))
        
        for file in os.listdir(component_path):
            vertices = []
            file_path = os.path.join(component_path, file)
            
            with open(file_path, 'r') as f:
                for line in f:
                    if line[0] == 'v':
                        _, x, y, z = line.split()
                        vertices.append([x, y, z])
            
            labels = np.append(labels, np.array([[label_now]]))
            label_test.append(vertices)
    
    label_test = np.vstack(label_test).reshape(-1, 1024, 3)
    label_test = label_test.astype("float64")
    
    with h5py.File("label_test-3.h5", 'w') as h5f:
        h5f.create_dataset('model_test-2', data=label_test)
        h5f.create_dataset('label_test-2', data=labels)
    
    return 0, 0

def rotate_point_cloud(pc):
    """
    Apply random rotation to point cloud around Y axis.
    
    Args:
        pc (numpy.ndarray): Input point cloud
        
    Returns:
        numpy.ndarray: Rotated point cloud
    """
    rotate_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotate_angle)
    sinval = np.sin(rotate_angle)
    rotation_matrix = np.array([
        [cosval, 0, sinval],
        [0, 1, 0],
        [-sinval, 0, cosval]
    ])
    return np.clip(np.dot(pc.reshape((-1, 3)), rotation_matrix), 0.0, 1.0)

def translate_pointcloud(pointcloud):
    """
    Apply random translation to point cloud.
    
    Args:
        pointcloud (numpy.ndarray): Input point cloud
        
    Returns:
        numpy.ndarray: Translated point cloud
    """
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    """
    Add random jitter to point cloud.
    
    Args:
        pointcloud (numpy.ndarray): Input point cloud
        sigma (float): Standard deviation of Gaussian noise
        clip (float): Maximum absolute value of noise
        
    Returns:
        numpy.ndarray: Jittered point cloud
    """
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

class MCB(Dataset):
    """
    Mechanical Components Binary (MCB) dataset class.
    
    Args:
        num_points (int): Number of points to use from each point cloud
        partition (str): Either 'test' or 'train'
    """
    def __init__(self, num_points, partition='test'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        """
        Get a single item from the dataset.
        
        Args:
            item (int): Index of the item to get
            
        Returns:
            tuple: (point cloud, label)
        """
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        
        if self.partition == 'train':
            pointcloud = rotate_point_cloud(pointcloud)
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        
        return pointcloud, label

    def __len__(self):
        """Get the total number of items in the dataset."""
        return self.data.shape[0]

if __name__ == '__main__':
    gen_data('test')
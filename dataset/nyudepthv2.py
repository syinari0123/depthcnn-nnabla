import os
import h5py
import numpy as np
from nnabla.utils.data_iterator import data_iterator
from nnabla.utils.data_source import DataSource
from nnabla.utils.data_source import SlicedDataSource

from . import transforms

color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
ORIG_HEIGHT, ORIG_WIDTH = 480, 640


def get_slice_start_end(size, n_slices, rank):
    _size = size // n_slices
    amount = size % n_slices
    slice_start = _size * rank
    if rank < amount:
        slice_start += rank
    else:
        slice_start += amount

    slice_end = slice_start + _size
    if slice_end > size:
        slice_start -= (slice_end - size)
        slice_end = size

    return slice_start, slice_end


def _get_sliced_data_source(ds, comm, shuffle=True):
    if comm is not None and comm.n_procs > 1:
        start, end = get_slice_start_end(ds._size, comm.n_procs, comm.rank)
        ds = SlicedDataSource(ds, shuffle=shuffle,
                              slice_start=start, slice_end=end)

    return ds


class NYUDepthIterator(DataSource):
    def __init__(self, root_dir, data_type, image_shape=(228, 304), shuffle=True, rng=None):
        super(NYUDepthIterator, self).__init__(shuffle=shuffle, rng=rng)

        self._data_list = self._get_nyu_datalist(root_dir, data_type)
        self._image_shape = image_shape
        self._size = len(self._data_list)
        self._variables = ("image", "depth")
        # Transform (including data augmentation)
        if data_type == 'train':
            self.transform = self.train_transform
        elif data_type in ['val', 'test']:
            self.transform = self.val_transform
        # Normalize
        self.mean_imagenet = np.asarray([0.485, 0.456, 0.406]).astype(np.float32).reshape(3, 1, 1)
        self.std_imagenet = np.asarray([0.229, 0.224, 0.225]).astype(np.float32).reshape(3, 1, 1)

        self.reset()

    def _h5_reader(self, path):
        """Image/Depth extractor from h5 format file.

        Args:
            path (str): Path to h5 format file
        Returns:
            rgb (np.uint8): RGB image (shape=[H,W,3])
            depth (np.float32): Depth image (shape=[H,W])
        """
        h5f = h5py.File(path, 'r')
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        return rgb, depth

    def _get_nyu_datalist(self, data_root, data_type, split_root='dataset/nyu_split'):
        """Load split data list from txt-file.
        Args:
            data_root: Path to nyudepth v2 dataset
            split_root: Choose from ['train','val','test']
            split_path: (Default:./nyu_split/*.txt)
        """
        assert data_type in ['train', 'val', 'test']
        # Load split
        f = open(os.path.join(split_root, '{}.txt'.format(data_type)), 'r')
        scene_ids = f.readlines()

        # Check whether each file exists in loaded list
        images = []
        data_root = os.path.expanduser(data_root)
        for s_id in scene_ids:
            d = os.path.join(data_root, s_id.rstrip(os.linesep))
            if os.path.exists(d) and d.endswith('.h5'):
                images.append(d)
        assert len(images) > 0, "Please check whether {} is correct data path.".format(data_root)

        return images

    def train_transform(self, rgb, depth):
        """
        Reference:
            https://github.com/fangchangma/sparse-to-dense.pytorch/blob/master/dataloaders/nyu_dataloader.py

        Args:
            rgb (np.uint8): RGB image (shape=[H,W,3])
            depth (np.float32): Depth image (shape=[H,W])

        Returns:
            rgb_np (np.float32): Tranformed RGB image
            depth_np (np.float32): Transformed Depth image
        """
        # Parameters for each augmentation
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # Perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / ORIG_HEIGHT),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self._image_shape),
            transforms.HorizontalFlip(do_flip)
        ])

        # Apply this transform to rgb/depth
        rgb = color_jitter(transform(rgb))  # random color jittering
        depth = transform(depth)

        # Normalize RGB
        rgb = np.asfarray(np.rollaxis(rgb, 2)) / 255.
        #rgb -= self.mean_imagenet
        # rgb /= self.std_imagenet  # [3,H,W]
        depth = depth[np.newaxis, :, :]  # [1,H,W]

        return rgb, depth

    def val_transform(self, rgb, depth):
        """
        Reference:
            https://github.com/fangchangma/sparse-to-dense.pytorch/blob/master/dataloaders/nyu_dataloader.py

        Args:
            rgb (np.uint8): RGB image (shape=[H,W,3])
            depth (np.float32): Depth image (shape=[H,W])

        Returns:
            rgb_np (np.float32): Tranformed RGB image
            depth_np (np.float32): Transformed Depth image
        """
        transform = transforms.Compose([
            transforms.Resize(240.0 / ORIG_HEIGHT),
            transforms.CenterCrop(self._image_shape),
        ])

        # Apply this transform to rgb/depth
        rgb = transform(rgb)
        depth = transform(depth)

        # Normalize RGB
        rgb = np.asfarray(np.rollaxis(rgb, 2)) / 255.
        #rgb -= self.mean_imagenet
        # rgb /= self.std_imagenet  # [3,H,W]
        depth = depth[np.newaxis, :, :]  # [1,H,W]

        return rgb, depth

    def reset(self):
        self._idxs = self._rng.permutation(self._size) if self.shuffle else np.arange(self._size)
        super(NYUDepthIterator, self).reset()

    def __iter__(self):
        self.reset()
        return self

    def _get_data(self, position):
        i = self._idxs[position]
        rgb, depth = self._h5_reader(self._data_list[i])
        rgb, depth = self.transform(rgb, depth)
        return rgb, depth


def create_data_iterator(root_dir, data_type, image_shape, batch_size,
                         comm=None, shuffle=True, rng=None,
                         with_memory_cache=False, with_parallel=False,
                         with_file_cache=False):
    ds = NYUDepthIterator(root_dir, data_type, image_shape, shuffle=shuffle, rng=rng)
    ds = _get_sliced_data_source(ds, comm, shuffle=shuffle)

    data_itr = data_iterator(ds, batch_size, with_memory_cache, with_parallel, with_file_cache)
    data_size = int(ds._size / batch_size)
    return {'itr': data_itr, 'size': data_size}

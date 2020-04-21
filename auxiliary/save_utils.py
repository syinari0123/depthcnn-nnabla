import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_depth2colormap(np_depths, normalize=True):
    colormaps = []
    for b in range(np_depths.shape[0]):
        np_depth = np_depths[b, 0]
        # Valid mask
        valid_mask = np_depth > 0
        if len(np_depth[valid_mask]) > 0:
            # Scaling from d_min & d_max
            if normalize:
                d_min = np.min(np_depth[valid_mask])
                d_max = np.max(np_depth[valid_mask])
                depth_relative = ((np_depth - d_min) / (d_max - d_min)) * valid_mask
            else:
                depth_relative = np_depth
        else:
            depth_relative = np.zeros_like(np_depth)  # this means all the values are zero

        # Output colored depthmap
        color_depth = plt.cm.jet(depth_relative)[:, :, :3].transpose(2, 0, 1)
        colormaps.append(color_depth)

    return np.stack(colormaps)

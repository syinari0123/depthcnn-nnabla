import numpy as np
import nnabla as nn
import nnabla.functions as F

from . import encoder_resnet50
from . import depth_decoder


def depth_cnn_model(image, test=False):
    """U-Net like architecture based on SharpNet [ICCVW'19]
    """
    # Normalization (imagenet)
    mean_imagenet = np.asarray([0.485, 0.456, 0.406]).astype(np.float32).reshape(1, 3, 1, 1)
    std_imagenet = np.asarray([0.229, 0.224, 0.225]).astype(np.float32).reshape(1, 3, 1, 1)
    var_mean = nn.Variable.from_numpy_array(mean_imagenet)
    var_std = nn.Variable.from_numpy_array(std_imagenet)
    image = (image - var_mean) / var_std

    # Encoder
    encoded_feats = encoder_resnet50(image, test=test)
    # Decoder
    pred_depth = depth_decoder(encoded_feats, test=test)
    return pred_depth

import os
import time
import argparse
import numpy as np
import cv2
from datetime import datetime

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.logger as logger
import nnabla.utils.save as save
from nnabla.monitor import Monitor, MonitorSeries, MonitorImageTile

from dataset import prepare_dataloader
from model import depth_cnn_model, l1_loss
from auxiliary import convert_depth2colormap


def main(args):
    from numpy.random import seed
    seed(46)

    # Get context.
    from nnabla.ext_utils import get_extension_context
    ctx = get_extension_context('cudnn', device_id='0', type_config='float')
    nn.set_default_context(ctx)

    # Create CNN network
    # Create input variables.
    image = nn.Variable([1, 3, args.img_height, args.img_width])
    label = nn.Variable([1, 1, args.img_height, args.img_width])
    # Create prediction graph.
    pred = depth_cnn_model(image, test=True)

    # Load pretrained model
    assert(args.pretrained_path != "")
    nn.parameter.load_parameters(args.pretrained_path)

    # Prepare monitors.
    monitor = Monitor(os.path.join(args.result_dir, 'nnmonitor'))
    monitors = {
        'test_viz': MonitorImageTile('Test images', monitor, interval=1, num_images=1)
    }

    # Initialize DataIterator
    data_dic = prepare_dataloader(args.dataset_path,
                                  datatype_list=['test'],
                                  batch_size=1,
                                  img_size=(args.img_height, args.img_width))

    # Inference loop.
    logger.info("Start inference!!!")
    for index in range(data_dic['test']['size']):
        logger.info("{}/{}".format(index, data_dic['test']['size']))

        image.d, label.d = data_dic['test']['itr'].next()
        pred.forward(clear_buffer=True)
        test_viz = np.concatenate([image.d,
                                   convert_depth2colormap(label.d),
                                   convert_depth2colormap(pred.d)], axis=3)
        monitors['test_viz'].add(index, test_viz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('depth-cnn-nnabla')
    parser.add_argument('--dataset-path', type=str, default="~/datasets/nyudepthv2")
    parser.add_argument('--pretrained-path', type=str, default="")
    parser.add_argument('--img-height', type=int, default=228)
    parser.add_argument('--img-width', type=int, default=304)
    parser.add_argument('--result-dir', default='result')
    args = parser.parse_args()
    main(args)

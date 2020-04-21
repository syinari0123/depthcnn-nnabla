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
    # === TRAIN ===
    # Create input variables.
    image = nn.Variable([args.batch_size, 3, args.img_height, args.img_width])
    label = nn.Variable([args.batch_size, 1, args.img_height, args.img_width])
    # Create prediction graph.
    pred = depth_cnn_model(image, test=False)
    pred.persistent = True
    # Create loss function.
    loss = l1_loss(pred, label)
    # === VAL ===
    #vimage = nn.Variable([args.batch_size, 3, args.img_height, args.img_width])
    #vlabel = nn.Variable([args.batch_size, 1, args.img_height, args.img_width])
    #vpred = depth_cnn_model(vimage, test=True)
    #vloss = l1_loss(vpred, vlabel)

    # Prepare monitors.
    monitor = Monitor(os.path.join(args.log_dir, 'nnmonitor'))
    monitors = {
        'train_epoch_loss': MonitorSeries('Train epoch loss', monitor, interval=1),
        'train_itr_loss': MonitorSeries('Train itr loss', monitor, interval=100),
        # 'val_epoch_loss': MonitorSeries('Val epoch loss', monitor, interval=1),
        'train_viz': MonitorImageTile('Train images', monitor, interval=1000, num_images=4)
    }

    # Create Solver. If training from checkpoint, load the info.
    if args.optimizer == "adam":
        solver = S.Adam(alpha=args.learning_rate, beta1=0.9, beta2=0.999)
    elif args.optimizer == "sgd":
        solver = S.Momentum(lr=args.learning_rate, momentum=0.9)
    solver.set_parameters(nn.get_parameters())

    # Initialize DataIterator
    data_dic = prepare_dataloader(args.dataset_path,
                                  datatype_list=['train', 'val'],
                                  batch_size=args.batch_size,
                                  img_size=(args.img_height, args.img_width))

    # Training loop.
    logger.info("Start training!!!")
    total_itr_index = 0
    for epoch in range(1, args.epochs + 1):
        ## === training === ##
        total_train_loss = 0
        index = 0
        while index < data_dic['train']['size']:
            # Preprocess
            image.d, label.d = data_dic['train']['itr'].next()
            loss.forward(clear_no_need_grad=True)
            # Initialize gradients
            solver.zero_grad()
            # Backward execution
            loss.backward(clear_buffer=True)
            # Update parameters by computed gradients
            if args.optimizer == 'sgd':
                solver.weight_decay(1e-4)
            solver.update()

            # Update log
            index += 1
            total_itr_index += 1
            total_train_loss += loss.d

            # Pass to monitor
            monitors['train_itr_loss'].add(total_itr_index, loss.d)

            # Visualization
            pred.forward(clear_buffer=True)
            train_viz = np.concatenate([image.d,
                                        convert_depth2colormap(label.d),
                                        convert_depth2colormap(pred.d)], axis=3)
            monitors['train_viz'].add(total_itr_index, train_viz)

            # Logger
            logger.info("[{}] {}/{} Train Loss {} ({})".format(epoch, index, data_dic['train']['size'],
                                                               total_train_loss / index, loss.d))

        # Pass training loss to a monitor.
        train_error = total_train_loss / data_dic['train']['size']
        monitors['train_epoch_loss'].add(epoch, train_error)

        # Save Parameter
        out_param_file = os.path.join(args.log_dir, 'checkpoint' + str(epoch) + '.h5')
        nn.save_parameters(out_param_file)

        ## === Validation === ##
        #total_val_loss = 0.0
        #val_index = 0
        # while val_index < data_dic['val']['size']:
        #    # Inference
        #    vimage.d, vlabel.d = data_dic['val']['itr'].next()
        #    vpred.forward(clear_buffer=True)
        #    vloss.forward(clear_buffer=True)
        #    total_val_loss += vloss.d
        #    val_index += 1
        #    break

        # Pass validation loss to a monitor.
        #val_error = total_val_loss / data_dic['val']['size']
        #monitors['val_epoch_loss'].add(epoch, val_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('depth-cnn-nnabla')
    parser.add_argument('--dataset-path', type=str, default="~/datasets/nyudepthv2")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-height', type=int, default=228)
    parser.add_argument('--img-width', type=int, default=304)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--log-dir', default='./log')
    args = parser.parse_args()
    main(args)

import nnabla.functions as F
import nnabla.parametric_functions as PF


def resize_feat(x, ref_size):
    assert (len(ref_size) == 2) and isinstance(ref_size, tuple)
    return F.interpolate(x, output_size=ref_size, mode='linear')


def upconv_block(x, num_conv, out_channel, kernel=3, bias=True, test=False, name=''):
    for i in range(num_conv):
        x = PF.convolution(x, out_channel, kernel=(kernel, kernel),
                           pad=(1, 1), stride=(1, 1), dilation=(1, 1), with_bias=bias,
                           name=name + '_conv_{}'.format(i))
        x = PF.batch_normalization(x, batch_stat=not test, name=name + '_bn_{}'.format(i))
        x = F.relu(x, inplace=True)
    return x


def depth_decoder(feat_list, num_layers=[6, 6, 2, 2, 2],
                  out_channels=[1024, 512, 256, 64, 16],
                  out_size=(228, 304), test=False):
    assert(len(feat_list) == 5)
    assert(len(num_layers) == len(out_channels))

    # upconv4
    x_out = resize_feat(feat_list[4], ref_size=tuple(feat_list[3].shape[2:]))
    x_out = upconv_block(x_out, num_layers[0], out_channels[0], test=test, name='upconv4')
    x_out = F.concatenate(x_out, feat_list[3], axis=1)

    # upconv3
    x_out = upconv_block(x_out, num_layers[1], out_channels[1], test=test, name='upconv3')
    x_out = resize_feat(x_out, ref_size=tuple(feat_list[2].shape[2:]))
    x_out = F.concatenate(x_out, feat_list[2], axis=1)

    # upconv2
    x_out = upconv_block(x_out, num_layers[2], out_channels[2], test=test, name='upconv2')
    x_out = resize_feat(x_out, ref_size=tuple(feat_list[1].shape[2:]))
    x_out = F.concatenate(x_out, feat_list[1], axis=1)

    # upconv1
    x_out = upconv_block(x_out, num_layers[3], out_channels[3], test=test, name='upconv1')
    x_out = resize_feat(x_out, ref_size=tuple(feat_list[0].shape[2:]))
    x_out = F.concatenate(x_out, feat_list[0], axis=1)

    # upconv0
    x_out = upconv_block(x_out, num_layers[4], out_channels[4], test=test, name='upconv0')
    x_out = resize_feat(x_out, ref_size=out_size)

    # final layer
    x_out = PF.convolution(x_out, 1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), dilation=(1, 1),
                           with_bias=True, name='final_conv')
    x_out = PF.batch_normalization(x_out, batch_stat=not test, name='final_bn')
    x_out = 10.0 * F.sigmoid(x_out) + 0.01  # better convergence?

    return x_out

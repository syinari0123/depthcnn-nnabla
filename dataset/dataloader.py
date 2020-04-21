import nnabla.logger as logger

from . import create_data_iterator


def prepare_dataloader(dataset_path,
                       datatype_list=['train', 'val', 'test'],
                       batch_size=8,
                       img_size=(228, 304)):
    assert all([dtype in ['train', 'val', 'test'] for dtype in datatype_list])

    data_dic = {'train': None, 'val': None, 'test': None}
    if 'train' in datatype_list:
        data_dic['train'] = create_data_iterator(dataset_path, data_type='train',
                                                 image_shape=img_size, batch_size=batch_size)
    if 'val' in datatype_list:
        data_dic['val'] = create_data_iterator(dataset_path, data_type='val',
                                               image_shape=img_size, batch_size=1, shuffle=False)
    if 'test' in datatype_list:
        data_dic['test'] = create_data_iterator(dataset_path, data_type='test',
                                                image_shape=img_size, batch_size=1, shuffle=False)

    # Dataset size information
    dataset_info = "> Dataset size: "
    for dtype in datatype_list:
        dataset_info += "[{}] {} ".format(dtype, data_dic[dtype]['size'])
    logger.info(dataset_info)

    return data_dic

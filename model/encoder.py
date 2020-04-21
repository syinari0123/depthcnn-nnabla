import os
import nnabla as nn
from nnabla.models.imagenet import ResNet50


class JudgeResidualFinal(object):
    def __init__(self):
        self.add2_flag = False
        self.relu_flag = False
        self.layer_final_flag = False

    def _reset(self):
        self.add2_flag = False
        self.relu_flag = False

    def __call__(self, layer_name):
        # The end of Residual block shoule be Add2 -> ReLU
        if layer_name == "Add2":
            self.add2_flag = True
        elif (self.add2_flag and layer_name == "ReLU"):
            self.relu_flag = True
        else:
            self.add2_flag = False

        # Judge this is the end or not
        if (self.add2_flag and self.relu_flag):
            self._reset()
            return True
        else:
            return False


class VisitFeatures(object):
    def __init__(self):
        self.features = []
        self.layer_judger = JudgeResidualFinal()
        self.layer_counter = 0
        self.features_at = set([3, 7, 13, 16])  # num_layers=[3,4,6,3]

    def __call__(self, f):
        # Extract Layer name
        crnt_layer = None
        if (f.name.startswith('ReLU')):
            crnt_layer = 'ReLU'
        elif f.name.startswith('Add2'):
            crnt_layer = 'Add2'

        # Judge wether this layer is the end of the redisual block or not
        if self.layer_judger(crnt_layer):
            self.layer_counter += 1
            if self.layer_counter in self.features_at:
                self.features.append(f.outputs[0])

        # + Maxpooling (0th)
        if (f.name.startswith('MaxPooling')):
            self.features.append(f.outputs[0])


def encoder_resnet50(x, test=False):
    """Extract mid feature of ResNet50 (outputs of each residual block)
    Shape of Each feature is following (when input_image is [B,3,228,304])
        - [B,64,57,76]
        - [B,256,57,76]
        - [B,512,29,38]
        - [B,1024,15,19]
        - [B,2048,8,10]
    """
    res50 = ResNet50()
    o = res50(x, use_up_to='lastconv+relu', training=not(test))
    f = VisitFeatures()
    o.visit(f)
    return f.features

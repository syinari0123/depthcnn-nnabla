import numbers
import types
import numpy as np
from PIL import Image, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import scipy.ndimage.interpolation as itpl


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


class Lambda:
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Resize:
    """Resize the the given ``numpy.ndarray`` to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.NEAREST):
        assert isinstance(size, int) or isinstance(size, float) or \
            (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = Image.fromarray(img)
        if isinstance(self.size, int) or isinstance(self.size, float):
            size = tuple((np.array(img.size) * self.size).astype(int))
        else:
            size = self.size
        return np.array(img.resize(size, self.interpolation))


class Rotate:
    """Rotates the given ``numpy.ndarray``.
    Args:
        angle (float): The rotation angle in degrees.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be rotated.
        Returns:
            img (numpy.ndarray (C x H x W)): Rotated image.
        """

        # order=0 means nearest-neighbor type interpolation
        return itpl.rotate(img, self.angle, reshape=False, prefilter=False, order=0)


class CenterCrop:
    """Crops the given ``numpy.ndarray`` at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for center crop.
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
        """
        h = img.shape[0]
        w = img.shape[1]
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        # # randomized cropping
        # i = np.random.randint(i-3, i+4)
        # j = np.random.randint(j-3, j+4)

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be cropped.

        Returns:
            img (numpy.ndarray (C x H x W)): Cropped image.
        """
        i, j, h, w = self.get_params(img, self.size)

        """
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))
        if img.ndim == 3:
            return img[i:i + h, j:j + w, :]
        elif img.ndim == 2:
            return img[i:i + h, j:j + w]
        else:
            raise RuntimeError('img should be ndarray with 2 or 3 dimensions. Got {}'.format(img.ndim))


class HorizontalFlip:
    """Horizontally flip the given ``numpy.ndarray``.
    Args:
        do_flip (boolean): whether or not do horizontal flip.
    """

    def __init__(self, do_flip):
        self.do_flip = do_flip

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Image to be flipped.

        Returns:
            img (numpy.ndarray (C x H x W)): flipped image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        if self.do_flip:
            return np.fliplr(img)
        else:
            return img


class ColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (numpy.ndarray (C x H x W)): Input image.

        Returns:
            img (numpy.ndarray (C x H x W)): Color jittered image.
        """
        if not(_is_numpy_image(img)):
            raise TypeError('img should be ndarray. Got {}'.format(type(img)))

        pil = Image.fromarray(img)
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return np.array(transform(pil))


class Compose:
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

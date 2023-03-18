from PIL import Image
from PIL import ImageFile
from io import BytesIO
import numpy as np
import albumentations as A
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300


class FeatureExtraction:
    """
    to return features of images or normalized image numpy array
    """
    def __init__(self, image_file):
        self.image_file = image_file
        self.normalize = A.Compose([A.Normalize(always_apply=True)])

    def get_features(self):
        image = Image.open(BytesIO(self.image_file))
        # resize the image
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image = np.array(image)
        image = self.normalize(image=image)['image']
        # model was trained in pytorch
        # in pytorch, the input image array should be in format NCHW
        # batch N, channels C, height H, width W.
        return np.transpose(image, (2, 0, 1)).astype(np.float32)

import numpy as np

from tensorflow.keras.preprocessing import image

def load_image(
        path:str,
        img_shape:tuple=None,
    ):
        img = image.load_img(path, target_size=img_shape)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img
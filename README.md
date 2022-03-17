# VGGFace

Oxford VGGFace Implementation updated to tensorflow v2

---

## How to install

To install the package, run the following command:
```
pip install git+https://github.com/claudiourbina/VGGFace
```

As soon as possible, it will be available in the [pypi](https://pypi.org/project/VGGFace/) repository.

# How to use

It is very easy to use the package. The following example shows how to use the package to extract the VGGFace features from a single image:

```python	
from vggface import VGGFace
from vggface.utils import load_image

model = VGGFace(
    architecture="resnet50", 
    include_top=False, 
    input_shape=(224, 224, 3)
)

im = load_image(path='face.png', img_shape=(224, 224, 3))
features = model.predict(im)
```

To make predictions:

```python	
from vggface import VGGFace
from vggface.utils import load_image

model = VGGFace(
    architecture="resnet50", 
    include_top=True, 
    input_shape=(224, 224, 3)
)

im = load_image(path='face.png', img_shape=(224, 224, 3))
pred = model.predict(im, postprocessing=True)
```

---

This repository is an updated version of VGGFace, based on [Oxford VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) and [keras-vggface](https://github.com/rcmalli/keras-vggface) by rcmalli.

---

## License

- [Oxford VGGFace](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
    `"These models can be used for non-commercial research purposes under Creative Commons Attribution License."`
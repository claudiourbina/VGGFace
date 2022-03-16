from .resnet50 import ResNet50

class VGGFace:
    """
    VGGFace

    VGGFace is a face recognition model.
    """
    def __init__(
        self,
        architecture:str="resnet50",
        include_top:bool=True,
        input_shape=None,
        pooling:bool=None,
        classes:int=8631
    ) -> None:
        """
        Initialize VGGFace

        :param architecture: The architecture of the VGGFace model.
        :param include_top: Whether to include the top layer.
        :param input_shape: The input shape of the VGGFace model.
        :param pooling: The pooling layer of the VGGFace model.
        :param classes: The number of classes of the VGGFace model.
        """
        self._input_shape = input_shape

        if architecture == "resnet50":
            self.model = ResNet50(
                include_top=include_top,
                input_shape=self._input_shape,
                pooling=pooling,
                classes=classes
            )

    def predict(
        self,
        x,
        postprocessing:bool=False
    ):
        """
        Predict

        :param x: The input of the VGGFace model.
        :param postprocessing: Whether to postprocessing the result.
            if True, the result will be the label and the probability.
            if False, the result will be the raw result.

        :return: The result of the VGGFace model.
        """
        return self.model.predict(x, postprocessing=postprocessing)
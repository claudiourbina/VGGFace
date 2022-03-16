from .resnet50 import ResNet50

class VGGFace:
    def __init__(
        self,
        architecture:str="resnet50",
        include_top:bool=True,
        input_shape=None,
        pooling:bool=None,
        classes:int=8631
    ) -> None:

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
        return self.model.predict(x, postprocessing=postprocessing)
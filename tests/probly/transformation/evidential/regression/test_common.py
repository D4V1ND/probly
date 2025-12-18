from __future__ import annotations

from probly.predictor import Predictor, predict
from probly.transformation.evidential.regression.common import evidential_regression, register


def test_predict_method(dummy_predictor: Predictor) -> None:
    """Test if evidential_regression returns an object with a predict method."""

    def simple_generator(base: Predictor) -> object:
        class Wrapper:
            def predict(self, x: object) -> object:
                return predict(base, x)

        return Wrapper()

    register(Predictor, simple_generator)

    model = evidential_regression(dummy_predictor)

    assert model is not None
    assert hasattr(model, "predict")

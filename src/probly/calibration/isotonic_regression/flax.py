import jax.numpy as jnp
from probly.calibration.template import CalibratorBaseFlax
from jax import Array

class IsotonicRegression(CalibratorBaseFlax):
    """Calibrator that uses isotonic regression."""

    def __init__(self):
        super().__init__()
        self.bin_boundaries : Array | None = None
        self.bin_probs : Array | None = None
        self.is_fitted = False

    def fit(self, y_pred: Array, y_binary: Array) -> IsotonicRegression: 
        
        self.bin_probs = _pav(y_pred, y_binary)

        boundaries_list = []
        for i in range(1, len(self.bin_probs)):
            if self.bin_probs[i] != self.bin_probs[i-1]:
                boundaries_list.append(y_pred[i])
        self.bin_boundaries = jnp.array(boundaries_list)
        self.is_fitted = True
        return self

    def predict(self, predictions: Array) -> Array:

        if not self.is_fitted:
            msg = "Calibrator must be fitted before Calibration"
            raise ValueError(msg)

        calibrated = []

        for pred in predictions:
            bin_id = jnp.searchsorted(self.bin_boundaries, pred)
            calibrated.append(jnp.unique(self.bin_probs)[bin_id])

        return jnp.array(calibrated)
    
    def _pav(x: Array, y: Array) -> Array: 

        xy = jnp.array([x,y])
        ind = jnp.argsort( xy[0,:] ); xy_xsorted = xy[:,ind]
        if jnp.all(xy_xsorted[1][:-1] <= xy_xsorted[1][1:]):
            return xy_xsorted[1]
        for i in range(1,len(x)):
            if xy_xsorted[1,i] < xy_xsorted[1,i-1]:
                mean = (xy_xsorted[1,i] + xy_xsorted[1,i-1])/2
                xy_xsorted = xy_xsorted.at[1,i].set(mean)
                xy_xsorted = xy_xsorted.at[1,i-1].set(mean)
        return _pav(xy_xsorted[0], xy_xsorted[1])
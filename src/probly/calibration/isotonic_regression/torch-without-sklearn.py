import torch
from torch import Tensor
from probly.calibration.template import CalibratorBaseTorch


class IsotonicRegression(CalibratorBaseTorch):
    """Calibrator that uses isotonic regression."""

    def __init__(self):
        super().__init__()
        self.bin_boundaries : Tensor | None = None
        self.bin_probs : Tensor | None = None
        self.is_fitted = False

    def fit(self, y_pred: Tensor, y_binary: Tensor) -> IsotonicRegression: 
        
        self.bin_probs = _pav(y_pred, y_binary)

        boundaries_list = []
        for i in range(1, len(self.bin_probs)):
            if self.bin_probs[i] != self.bin_probs[i-1]:
                boundaries_list.append(y_pred[i])
        self.bin_boundaries = torch.tensor(boundaries_list)
        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:

        if not self.is_fitted:
            msg = "Calibrator must be fitted before Calibration"
            raise ValueError(msg)

        calibrated = []

        for pred in predictions:
            bin_id = torch.searchsorted(self.bin_boundaries, pred)
            calibrated.append(torch.unique(self.bin_probs)[bin_id])

        return torch.tensor(calibrated)
    
    def _pav(x: Tensor, y: Tensor) -> Tensor: 

        xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), axis=0)
        ind = torch.argsort( xy[0,:] ); xy_xsorted = xy[:,ind]
        if torch.all(xy_xsorted[1][:-1] <= xy_xsorted[1][1:]):
            return xy_xsorted[1]
        for i in range(1,len(x)):
            if xy_xsorted[1,i] < xy_xsorted[1,i-1]:
                xy_xsorted[1,i] = xy_xsorted[1,i-1] = (xy_xsorted[1,i] + xy_xsorted[1,i-1])/2
        return _pav(xy_xsorted[0], xy_xsorted[1])
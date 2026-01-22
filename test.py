from __future__ import annotations  # noqa: D100

import torch

num_bins = 4
calibration_set = torch.tensor(
    [0, 0.1, 0.2, 0.25, 0.26, 0.29, 0.3, 0.4,
     0.5, 0.6, 0.7, 0.15, 0.9, 0.99, 1],
    dtype=torch.float32,
)

edges = torch.quantile(
    calibration_set,
    torch.linspace(0, 1, num_bins + 1),
)
edges[0] = 0.0
edges[-1] = 1.0
print(edges)

bin_ids = torch.bucketize(calibration_set, edges) - 1
bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)
print(bin_ids)

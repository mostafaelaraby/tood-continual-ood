# Third-Party Software Notices

The root [`LICENSE`](LICENSE) applies to the original TOOD code and
documentation in this repository. It does not replace licenses or copyright
notices that apply to third-party or vendored material.

## OpenOOD

The `openood/` directory and OpenOOD-derived configuration files under
`configs/datasets/`, `configs/networks/`, `configs/pipelines/`,
`configs/postprocessors/`, and `configs/preprocessors/` contain material
vendored from:

- Project: OpenOOD
- Source: <https://github.com/Jingkang50/OpenOOD>
- Copyright: Copyright (c) 2021 Jingkang Yang
- License: MIT
- Preserved license: [`openood/LICENSE`](openood/LICENSE)

Local modifications and TOOD integration code do not remove or replace the
upstream OpenOOD notice.

## ODIN

`openood/postprocessors/odin_postprocessor.py` identifies itself as adapted
from:

- Project: ODIN
- Source: <https://github.com/facebookresearch/odin>
- Copyright: Copyright (c) 2017-present, Facebook, Inc.
- Upstream license: Creative Commons Attribution-NonCommercial 4.0
  International (`CC-BY-NC-4.0`)
- License notice: [`LICENSES/ODIN-CC-BY-NC-4.0.md`](LICENSES/ODIN-CC-BY-NC-4.0.md)

To the extent the local implementation contains adapted ODIN material, its
use is limited to noncommercial purposes by that license. It is not covered
by the repository's root MIT grant. Anyone intending commercial use should
replace this component with an independently implemented alternative or
obtain appropriate permission.

## PyTorch Classification ResNet

`openood/networks/temp.py` identifies itself as adapted from:

- Project: `bearpaw/pytorch-classification`
- Source: <https://github.com/bearpaw/pytorch-classification>
- Copyright: Copyright (c) 2017 Wei Yang
- License: MIT
- Preserved license:
  [`LICENSES/pytorch-classification-MIT.txt`](LICENSES/pytorch-classification-MIT.txt)

## Apache-2.0 material

The following vendored files retain Apache-2.0 copyright or license notices:

- `openood/utils/comm.py` and `openood/utils/launch.py`: Facebook-authored
  distributed-training utilities, modified for OpenOOD/TOOD integration.
- `openood/postprocessors/patchcore_postprocessor.py`: portions marked
  Copyright 2017 Google Inc., modified for OpenOOD/TOOD integration.

The full Apache License 2.0 is preserved at
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Other source acknowledgements

Several OpenOOD files contain links to research implementations or papers
from which algorithms were adapted or referenced. Those inline notices are
retained. The notices above cover the third-party licensing markers found in
the vendored source during this release audit; they do not claim ownership of
third-party trademarks, datasets, model weights, or publications.

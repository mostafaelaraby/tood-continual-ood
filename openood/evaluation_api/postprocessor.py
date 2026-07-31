import os
import urllib.request

from openood.postprocessors import (
    AdaScalePostprocessor,
    ASHPostprocessor,
    BasePostprocessor,
    CIDERPostprocessor,
    ConfBranchPostprocessor,
    CosinePostProcessor,
    CutPastePostprocessor,
    DICEPostprocessor,
    DRAEMPostprocessor,
    DropoutPostProcessor,
    DSVDDPostprocessor,
    EBOPostprocessor,
    EnsemblePostprocessor,
    GaiaPostprocessor,
    GENPostprocessor,
    GMMPostprocessor,
    GodinPostprocessor,
    GradNormPostprocessor,
    GRAMPostprocessor,
    GrOODPostprocessor,
    KLMatchingPostprocessor,
    KNNPostprocessor,
    MaxLogitPostprocessor,
    MCDPostprocessor,
    MDSEnsemblePostprocessor,
    MDSPostprocessor,
    MOSPostprocessor,
    NNGuidePostprocessor,
    NPOSPostprocessor,
    ODINPostprocessor,
    OpenGanPostprocessor,
    OpenMax,
    PatchcorePostprocessor,
    RankFeatPostprocessor,
    Rd4adPostprocessor,
    ReactPostprocessor,
    ResidualPostprocessor,
    RMDSPostprocessor,
    RotPredPostprocessor,
    SHEPostprocessor,
    SSDPostprocessor,
    TemperatureScalingPostprocessor,
    VIMPostprocessor,
)
from openood.utils.config import Config, merge_configs

postprocessors = {
    "ash": ASHPostprocessor,
    "cider": CIDERPostprocessor,
    "conf_branch": ConfBranchPostprocessor,
    "msp": BasePostprocessor,
    "ebo": EBOPostprocessor,
    "odin": ODINPostprocessor,
    "mds": MDSPostprocessor,
    "mds_ensemble": MDSEnsemblePostprocessor,
    "npos": NPOSPostprocessor,
    "rmds": RMDSPostprocessor,
    "gmm": GMMPostprocessor,
    "grood": GrOODPostprocessor,
    "cosine": CosinePostProcessor,
    "patchcore": PatchcorePostprocessor,
    "openmax": OpenMax,
    "react": ReactPostprocessor,
    "vim": VIMPostprocessor,
    "gradnorm": GradNormPostprocessor,
    "godin": GodinPostprocessor,
    "mds": MDSPostprocessor,
    "gram": GRAMPostprocessor,
    "cutpaste": CutPastePostprocessor,
    "mls": MaxLogitPostprocessor,
    "residual": ResidualPostprocessor,
    "klm": KLMatchingPostprocessor,
    "temp_scaling": TemperatureScalingPostprocessor,
    "ensemble": EnsemblePostprocessor,
    "dropout": DropoutPostProcessor,
    "draem": DRAEMPostprocessor,
    "dsvdd": DSVDDPostprocessor,
    "mos": MOSPostprocessor,
    "mcd": MCDPostprocessor,
    "opengan": OpenGanPostprocessor,
    "knn": KNNPostprocessor,
    "dice": DICEPostprocessor,
    "ssd": SSDPostprocessor,
    "she": SHEPostprocessor,
    "rd4ad": Rd4adPostprocessor,
    "rotpred": RotPredPostprocessor,
    "rankfeat": RankFeatPostprocessor,
    "gen": GENPostprocessor,
    "gaia": GaiaPostprocessor,
    "adascale_a": AdaScalePostprocessor,
    "adascale_l": AdaScalePostprocessor,
    "nnguide": NNGuidePostprocessor,
}

link_prefix = "https://raw.githubusercontent.com/Jingkang50/OpenOOD/main/configs/postprocessors/"


def get_postprocessor(
    config_root: str, postprocessor_name: str, id_data_name: str
):
    postprocessor_config_path = os.path.join(
        config_root, "postprocessors", f"{postprocessor_name}.yml"
    )
    if not os.path.exists(postprocessor_config_path):
        os.makedirs(os.path.dirname(postprocessor_config_path), exist_ok=True)
        urllib.request.urlretrieve(
            link_prefix + f"{postprocessor_name}.yml", postprocessor_config_path
        )

    config = Config(postprocessor_config_path)
    config = merge_configs(
        config, Config(**{"dataset": {"name": id_data_name}})
    )
    postprocessor = postprocessors[postprocessor_name](config)
    postprocessor.APS_mode = config.postprocessor.APS_mode
    postprocessor.hyperparam_search_done = False
    return postprocessor

import torch.utils.checkpoint as cp
from mmcls.models.backbones import TIMMBackbone
from mmcls.registry import MODELS
from timm.models.efficientnet import default_cfgs


@MODELS.register_module(force=True)
class TimmEfficientNet(TIMMBackbone):
    """support torch.utils.checkpoint."""
    def __init__(self, model_name, with_cp=False, **kwargs):
        assert model_name in default_cfgs, \
            f'timm.models.efficientnet does not have {model_name}.'
        super().__init__(model_name=model_name, **kwargs)
        self.with_cp = with_cp

    def forward(self, x):
        x = self.timm_model.conv_stem(x)
        x = self.timm_model.bn1(x)
        x = self.timm_model.act1(x)
        for blocks in self.timm_model.blocks:
            for block in blocks:
                if self.with_cp and x.requires_grad:
                    x = cp.checkpoint(block, x)
                else:
                    x = block(x)
        x = self.timm_model.conv_head(x)
        x = self.timm_model.bn2(x)
        feats = self.timm_model.act2(x)
        return (feats,)

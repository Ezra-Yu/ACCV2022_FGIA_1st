from typing import List

from mmdet.models.builder import MODELS

from mmengine.model import BaseTestTimeAugModel


@MODELS.register_module()
class ClassifierTTA(BaseTestTimeAugModel):
    pass
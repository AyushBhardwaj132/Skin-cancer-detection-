import torch
from src.models.fusion_model import FusionModel
from src.models.model import GeM


def test_gem_pooling():
    gem = GeM(p=3.0)
    x = torch.randn(2, 64, 16, 16)
    out = gem(x)
    assert out.shape == (2, 64, 1, 1)


def test_fusion_model_forward():
    model = FusionModel(backbone_name="tf_efficientnetv2_m", metadata_dim=47, pretrained=False)
    img = torch.randn(2, 3, 384, 384)
    meta = torch.randn(2, 47)
    out = model(img, meta)
    assert out.shape == (2, 1)

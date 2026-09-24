"""Load an original DINO-WM checkpoint (gaoyuezhou/dino_wm `model_latest.pth`) as a
stable_worldmodel PreJEPA, so it plugs into the same CEM solver / WorldModelPolicy as LeWM.

The checkpoint pickles modules from the dino_wm repo (models.vit / models.proprio /
models.vqvae); their attribute names match stable_worldmodel's prejepa port, so unpickling
maps them onto those classes. The frozen encoder is not saved: DINOv2 ViT-S/14 from
torch.hub, `x_norm_patchtokens`, on images Normalize(0.5, 0.5) then Resize(196) (14x14
patches -- the predictor's pos_embedding is 3 x 196). Latent = [visual 384 | proprio 10 |
action 10] per patch, as in dino_wm's concat_dim=1. Planning cost = dino_wm's
objective_fn_last with its plan.yaml alpha=1: MSE(visual) + MSE(proprio emb).

Inputs must be normalised with DINO-WM's own Push-T stats (DINOWM_PUSHT_STATS), not the
LeWM dataset's, and pixels arrive ImageNet-normalised from the shared eval transform.
"""

import pickle
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2 as T

from stable_worldmodel.wm.prejepa import PreJEPA
from stable_worldmodel.wm.prejepa import module as M

# dino_wm datasets/pusht_dset.py precomputed stats (per env step; action tiled over frameskip)
DINOWM_PUSHT_STATS = dict(
    action_mean=np.array([-0.0087, 0.0068]), action_std=np.array([0.2019, 0.2002]),
    proprio_mean=np.array([236.6155, 264.5674, -2.93032027, 2.54307914]),
    proprio_std=np.array([101.1202, 87.0112, 74.84556075, 74.14009094]),
)

_MAP = {("models.vit", "ViTPredictor"): M.CausalPredictor, ("models.vit", "Transformer"): M.Transformer,
        ("models.vit", "Attention"): M.Attention, ("models.vit", "FeedForward"): M.FeedForward,
        ("models.proprio", "ProprioceptiveEmbedding"): M.Embedder}


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) in _MAP:
            return _MAP[(mod, name)]
        if mod.startswith("models."):
            return nn.Module          # decoder (unused for planning)
        if mod.startswith("accelerate"):
            return _Stub              # optimizer wrappers
        return super().find_class(mod, name)


class DinoV2Backbone(nn.Module):
    IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        self.base = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.resize = T.Resize(196)

    def forward(self, x, **_):
        x = x * self.IMNET_STD.to(x) + self.IMNET_MEAN.to(x)      # undo the shared eval transform
        x = self.resize((x - 0.5) / 0.5)                          # dino_wm default_transform + encoder_transform
        tok = self.base.forward_features(x)["x_norm_patchtokens"]
        tok = torch.cat([tok[:, :1], tok], 1)                     # dummy CLS: PreJEPA drops token 0
        return types.SimpleNamespace(last_hidden_state=tok)


class DinoWM(PreJEPA):
    predict_chunk = 50   # 300 CEM candidates x 588 tokens do not fit a 4 GB GPU in one pass

    def predict(self, embedding):
        return torch.cat([super(DinoWM, self).predict(e) for e in embedding.split(self.predict_chunk)])

    def get_cost(self, info_dict, action_candidates):
        device = next(self.parameters()).device
        for k in list(info_dict):
            if k.endswith("emb") or k.startswith("predicted_"):   # left by the previous CEM iteration
                del info_dict[k]
            elif torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)
        # PreJEPA caches start/goal encodings keyed on id/step_idx, which this policy doesn't pass
        for a in ("_init_cached_info", "_goal_cached_info"):
            self.__dict__.pop(a, None)
        return super().get_cost(info_dict, action_candidates.to(device))


def load_dinowm(ckpt_path, device="cuda"):
    pm = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="dinowm_pickle")
    ck = torch.load(str(ckpt_path), map_location="cpu", pickle_module=pm, weights_only=False)
    for mod in ck["predictor"].modules():   # dino_wm keeps the frame-causal mask as a plain attribute
        if isinstance(mod, M.Attention) and "bias" not in mod._buffers:
            mask = mod.__dict__.pop("bias")
            mod.register_buffer("bias", mask)
    model = DinoWM(encoder=DinoV2Backbone(), predictor=ck["predictor"],
                   extra_encoders=nn.ModuleDict({"proprio": ck["proprio_encoder"],
                                                 "action": ck["action_encoder"]}),
                   history_size=3, num_pred=1)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model


DEFAULT_CKPT = Path(__file__).resolve().parents[2] / "dinowm" / "outputs" / "pusht" / "checkpoints" / "model_latest.pth"

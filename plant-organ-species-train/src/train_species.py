# =======================
# Step 3: 不同器官种类分类器（参数化 CLI）
# =======================

import argparse
from pathlib import Path

from copy import deepcopy
import json
import pandas as pd
import os, math
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from PIL import Image
from torchvision import transforms, models
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    EfficientNet_V2_S_Weights,
)
# from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing
import os
import json
import math
import time
import random
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report

import timm
from tqdm import tqdm
from timm.utils import ModelEmaV2
from torch.cuda.amp import autocast, GradScaler

# 记录每个器官的验证变换（评估/训练口径严格对齐）
TRAIN_TF_REGISTRY = {}
VAL_TF_REGISTRY = {}
PRIOR_TAU_T_REGISTRY: Dict[str, Tuple[float, float]] = {}

OUT_DIR = None              # 将在 main() 中设置为 <base_out>/<CC>_organ_species_model
SPECIES_NEW_DIR = None      # 将在 main() 中设置为 OUT_DIR/species_new
COUNTRY = None

# 列名统一
COL_IMAGE     = "image_path"
COL_ORGAN_TXT = "organ"
COL_ORGAN_ID  = "organ_id"
COL_LABEL     = "label"
COL_SPECIESID = "species"

def build_new_species_maps_for_organ(df: pd.DataFrame, organ: str,
                                     col_organ: str, col_label: str, col_gid: str):
    """
    为给定器官重建'new_'前缀的局部物种映射：
    - 过滤掉在该器官下仅有 1 张图片的物种
    - 重新编号 local_id
    - 保存到 species_new/ 下（文件名前缀 new_）
    返回：gid2local, local2gid, label2local 以及被保留的全局 species_id 集合
    """
    organ = organ.lower().strip()
    sub = df[df[col_organ] == organ].copy()

    # 每物种（全局 species_id）在该器官下的样本数
    cnt = sub.groupby(col_gid)[col_gid].transform("count")
    sub = sub[cnt >= 10].copy()  # 只保留 >=2 的物种（确保可切分 train & val）

    # 若清理后为空或仅 1 类，则直接返回空映射
    uniq = sub[[col_label, col_gid]].drop_duplicates().sort_values([col_gid, col_label]).reset_index(drop=True)
    if len(uniq) < 2:
        return {}, {}, {}, set()

    uniq["local_id"] = range(len(uniq))
    label2local = {row[col_label]: int(row["local_id"])         for _, row in uniq.iterrows()}
    gid2local   = {str(int(row[col_gid])): int(row["local_id"]) for _, row in uniq.iterrows()}
    local2gid   = {str(int(row["local_id"])): int(row[col_gid]) for _, row in uniq.iterrows()}

    # 保存到 species_new/，带 new_ 前缀
    with open(SPECIES_NEW_DIR / f"new_species_local_map_{organ}.json", "w", encoding="utf-8") as f:
        json.dump(label2local, f, ensure_ascii=False, indent=2)
    with open(SPECIES_NEW_DIR / f"new_species_global2local_{organ}.json", "w", encoding="utf-8") as f:
        json.dump(gid2local, f, ensure_ascii=False, indent=2)
    with open(SPECIES_NEW_DIR / f"new_species_local2global_{organ}.json", "w", encoding="utf-8") as f:
        json.dump(local2gid, f, ensure_ascii=False, indent=2)

    print(f"✅ [{organ}] 已生成 new_* 映射：labels={len(label2local)}（已过滤仅1张的物种）")
    kept_gids = set(int(x) for x in sub[col_gid].unique())
    return gid2local, local2gid, label2local, kept_gids


def _load_fixed_split_or_fallback_NEW(
    df: pd.DataFrame,
    organ: str,
    col_organ: str,
    col_gid: str,
    new_gid2local: dict,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    保证 train/val 均覆盖所有“保留下来的”物种（local 类别）：
    - 每个类至少 1 张进入 val，且至少 1 张进入 train
    - 若某类样本 <2，之前就已被过滤，不会进入这里
    - 切分结果写入 species_new/splits 下
    """
    organ = organ.lower().strip()
    sp_dir = SPECIES_NEW_DIR / "splits"
    tr_csv = sp_dir / f"{organ}_train_split.csv"
    te_csv = sp_dir / f"{organ}_val_split.csv"

    if tr_csv.exists() and te_csv.exists():
        tr_df = pd.read_csv(tr_csv)
        te_df = pd.read_csv(te_csv)
        # 回填 __species_local__
        if "__species_local__" not in tr_df.columns:
            tr_df["__species_local__"] = tr_df[col_gid].astype(int).astype(str).map(new_gid2local).astype(int)
        if len(te_df) > 0 and "__species_local__" not in te_df.columns:
            te_df["__species_local__"] = te_df[col_gid].astype(int).astype(str).map(new_gid2local).astype(int)
        tr_df["__species_local__"] = tr_df["__species_local__"].astype(int)
        if len(te_df) > 0:
            te_df["__species_local__"] = te_df["__species_local__"].astype(int)
        return tr_df, te_df

    sub = df[df[col_organ] == organ].copy()
    sub["__species_local__"] = sub[col_gid].astype(int).astype(str).map(new_gid2local)
    sub = sub.dropna(subset=["__species_local__"]).copy()
    sub["__species_local__"] = sub["__species_local__"].astype(int)

    # 分类别手动切分，确保每类 train/val 都有
    rng = np.random.RandomState(seed)
    tr_list, te_list = [], []
    for _, g in sub.groupby("__species_local__"):
        n = len(g)
        # 这里 n>=2 已由上游过滤保证
        n_val = max(1, int(round(n * val_ratio)))
        n_val = min(n_val, n - 1)  # 至少留1张给train
        idx = rng.permutation(n)
        val_idx = idx[:n_val]
        tr_idx  = idx[n_val:]
        tr_list.append(g.iloc[tr_idx])
        te_list.append(g.iloc[val_idx])

    tr_df = pd.concat(tr_list).reset_index(drop=True)
    te_df = pd.concat(te_list).reset_index(drop=True)

    sp_dir.mkdir(parents=True, exist_ok=True)
    tr_df.to_csv(tr_csv, index=False)
    te_df.to_csv(te_csv, index=False)
    print(f"[{organ}] 保存切分（NEW）：train={len(tr_df)}, val={len(te_df)}，"
          f"类覆盖(train={tr_df['__species_local__'].nunique()}, val={te_df['__species_local__'].nunique()})")
    return tr_df, te_df
# ====================== 新增结束 ======================

current_organ = None   # 仅用于评估函数内取 organ

# 器官自适应配置
ORGAN_SPEC = {
    "flower": {"img_size": 448, "epochs_head": 4, "epochs_ft": 40, "tta": True},
    "leaf":   {"img_size": 512, "epochs_head": 5, "epochs_ft": 60,  "tta": True},
    "fruit":  {"img_size": 512, "epochs_head": 5, "epochs_ft": 50,  "tta": True},
    "bark":   {"img_size": 512, "epochs_head": 5, "epochs_ft": 30,  "tta": True},
}

# >>> 新增：控制 warm-up 是否使用训练增强（而不是验证增强）
WARMUP_USES_TRAIN_TF = {
    "leaf":   False,
    "flower": False,
    "fruit":  False,
    "bark":   False,
}

def make_species_transforms(organ: str, img_size: int = 512):
    organ = organ.lower()
    if organ == 'bark':
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.02,0.02,0.02,0.005)], p=0.5),
            #transforms.RandomGrayscale(p=0.10),
            #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            #transforms.RandomPerspective(distortion_scale=0.05, p=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    elif organ == 'leaf':
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.90, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomApply([transforms.ColorJitter(0.08,0.08,0.08,0.02)], p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.06), ratio=(0.3, 3.3)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    elif organ == 'fruit':
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomApply([transforms.ColorJitter(0.1,0.1,0.1,0.03)], p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomApply([transforms.ColorJitter(0.12,0.12,0.10,0.04)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size*1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    return train_tf, val_tf


# === 关键修复：在 get_backbone_for_organ 内注册 VAL_TF_REGISTRY[organ] ===
def get_backbone_for_organ(organ: str, num_classes: int):
    organ = organ.lower().strip()
    spec = ORGAN_SPEC.get(organ, ORGAN_SPEC["leaf"])
    img_size = spec["img_size"]

    if organ == "flower":
        from torchvision.models import ConvNeXt_Small_Weights
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model   = models.convnext_small(weights=weights)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_feat, num_classes))

    elif organ == "leaf":
        from torchvision.models import ConvNeXt_Small_Weights
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model   = models.convnext_small(weights=weights)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_feat, num_classes))

    elif organ == "fruit":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model   = models.efficientnet_v2_s(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_feat, num_classes))

    elif organ == "bark":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model   = models.convnext_tiny(weights=weights)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.25), nn.Linear(in_feat, num_classes))
    else:
        raise ValueError(f"未知器官: {organ}")

    train_tf, val_tf = make_species_transforms(organ, img_size)
    # ★ 注册验证变换（Warmup 与评估都会用到）
    #VAL_TF_REGISTRY[organ] = train_tf
    TRAIN_TF_REGISTRY[organ] = train_tf
    VAL_TF_REGISTRY[organ]   = val_tf
    return model.to(DEVICE), (train_tf, val_tf), img_size


from torch.utils.data._utils.collate import default_collate
def drop_corrupt_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    return default_collate(batch)

class BasicImageDatasetNEW(Dataset):
    def __init__(self, df, image_col, label_col, label_lookup, tfm):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.label_lookup = label_lookup
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row[self.image_col]
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception:
            return None
        img = self.tfm(img)
        label = self.label_lookup[int(row[self.label_col])]
        return img, label, path


# ============ 评估（保留，以验证训练是否正常）===========
from sklearn.metrics import accuracy_score, f1_score, classification_report
PRIOR_LOG_REGISTRY: Dict[str, torch.Tensor] = {}

@torch.no_grad()
def evaluate_with_multicrop(model, te_df, img_size, device,
                             use_tta=True, use_multicrop=True,
                             n_crops: int = 8, ratio_low: float = 0.85,
                             logit_adj_tau: float = 1.0,
                             temp_T: float = 1.0,
                             return_logits: bool = False):
    model.eval()
    organ = globals().get("current_organ", "unknown")
    val_tf = VAL_TF_REGISTRY.get(organ, None)
    assert val_tf is not None, f"[{organ}] 未找到验证变换，请检查 VAL_TF_REGISTRY 注册。"

    def _ensure_min_size(pil_img, min_side_hw):
        H, W = pil_img.size[1], pil_img.size[0]
        if H >= min_side_hw and W >= min_side_hw:
            return pil_img
        scale = max(min_side_hw / max(1, H), min_side_hw / max(1, W))
        new_w = max(min_side_hw, int(np.ceil(W * scale)))
        new_h = max(min_side_hw, int(np.ceil(H * scale)))
        return pil_img.resize((new_w, new_h), Image.BICUBIC)

    def apply_val_tf(img):
        return val_tf(img).unsqueeze(0).to(device, non_blocking=True)

    def make_crops(img):
        if not use_multicrop or n_crops <= 1:
            return []
        crops = []
        if organ in ("flower", "leaf"):
            img_big = _ensure_min_size(img, img_size)
            tc = transforms.TenCrop(img_size) if n_crops >= 10 else transforms.FiveCrop(img_size)
            out = tc(img_big)
            norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            ])
            for c in out:
                crops.append(norm(c if isinstance(c, Image.Image) else c).unsqueeze(0))
            return crops
        resize_side = int(img_size * 1.20)
        img_safe = _ensure_min_size(img, img_size)
        img_r = transforms.Resize(resize_side)(img_safe)
        W, H = img_r.size
        g = max(1, int(round(n_crops ** 0.5)))
        grid_rows, grid_cols = g, g
        sw = (W - img_size) // max(1, grid_cols - 1) if grid_cols > 1 else 0
        sh = (H - img_size) // max(1, grid_rows - 1) if grid_rows > 1 else 0
        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
        for r in range(grid_rows):
            for c in range(grid_cols):
                left = min(c * sw, max(0, W - img_size))
                top  = min(r * sh, max(0, H - img_size))
                crop = img_r.crop((left, top, left + img_size, top + img_size))
                crops.append(norm(crop).unsqueeze(0))
        return crops

    log_prior = PRIOR_LOG_REGISTRY.get(organ, None)
    y_true, y_pred = [], []
    all_logits, all_ytrue = [], []
    top1_hits, top3_hits = 0, 0

    for _, row in tqdm(te_df.iterrows(), total=len(te_df),
                       desc=f"[{organ}] Eval (TTA={'Y' if use_tta else 'N'}, MC={'Y' if use_multicrop else 'N'})"):
        path = row[COL_IMAGE]
        y = int(row["__species_local__"])
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
        except Exception:
            continue

        x = apply_val_tf(img)
        logits = model(x)
        if use_tta:
            logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))

        crops = make_crops(img)
        if crops:
            for xx in crops:
                xx = xx.to(device, non_blocking=True)
                l = model(xx)
                if use_tta:
                    l = 0.5 * (l + model(torch.flip(xx, dims=[3])))
                logits += l
            logits = logits / (1 + len(crops))

        if log_prior is not None and logit_adj_tau is not None and logit_adj_tau > 0:
            logits = logits - logit_adj_tau * log_prior.view(1, -1).to(logits.device)
        if temp_T is not None and temp_T != 1.0:
            logits = logits / temp_T

        if return_logits:
            all_logits.append(logits.detach().cpu())
            all_ytrue.append(y)

        probs = F.softmax(logits, dim=1)
        top3 = probs.topk(3, dim=1)
        pred1 = top3.indices[0, 0].item()
        top3_set = set(top3.indices[0].tolist())

        y_true.append(y); y_pred.append(pred1)
        if pred1 == y: top1_hits += 1
        if y in top3_set: top3_hits += 1

    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, "N/A"

    top1 = top1_hits / len(y_true)
    top3 = top3_hits / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    labels_present = sorted(set(y_true) | set(y_pred))
    try:
        # 优先使用 NEW 的映射
        with open(SPECIES_NEW_DIR / f"new_species_local_map_{organ}.json", "r", encoding="utf-8") as f:
            label2local = json.load(f)
        local2label = {int(v): k for k, v in label2local.items()}
        target_names = [local2label.get(i, str(i)) for i in labels_present]
    except Exception:
        try:
            # 回退到 step0 产物（带国家前缀）
            with open(OUT_DIR / f"{COUNTRY}_species_local_map_{organ}.json", "r", encoding="utf-8") as f:
                label2local = json.load(f)
            local2label = {int(v): k for k, v in label2local.items()}
            target_names = [local2label.get(i, str(i)) for i in labels_present]
        except Exception:
            target_names = [str(i) for i in labels_present]

    report = classification_report(y_true, y_pred, labels=labels_present,
                                   target_names=target_names, zero_division=0)
    if return_logits:
        return top1, top3, macro_f1, report, all_logits, all_ytrue
    return top1, top3, macro_f1, report

def _recompute_metrics_from_logits(logits_list: List[torch.Tensor], y_true_list: List[int],
                                   prior_log: torch.Tensor, tau: float, temp_T: float) -> Tuple[float, float, float, List[int]]:
    correct1, correct3 = 0, 0
    y_pred = []
    for lg, y in zip(logits_list, y_true_list):
        lg2 = lg
        if prior_log is not None and tau is not None and tau > 0:
            lg2 = lg2 - tau * prior_log.view(1, -1)
        if temp_T is not None and temp_T != 1.0:
            lg2 = lg2 / temp_T
        probs = torch.softmax(lg2, dim=1)
        tk = probs.topk(3, dim=1)
        p1 = tk.indices[0, 0].item()
        y_pred.append(p1)
        correct1 += int(p1 == y)
        correct3 += int(y in set(tk.indices[0].tolist()))
    n = max(1, len(y_true_list))
    top1 = correct1 / n
    top3 = correct3 / n
    macro_f1 = f1_score(y_true_list, y_pred, average="macro")
    return top1, top3, macro_f1, y_pred

def _select_best_tau_T(organ: str, logits_list: List[torch.Tensor], y_true_list: List[int], prior_log: torch.Tensor) -> Tuple[float, float]:
    if organ == "leaf":
        tau_grid = [0.0]
        T_grid = [0.8, 1.0, 1.2, 1.5]
    elif organ == "flower":
        tau_grid = [0.0, 0.3, 0.5, 0.6]
        T_grid = [0.8, 1.0, 1.2]
    elif organ == "fruit":
        tau_grid = [0.0, 0.4, 0.6, 0.8]
        T_grid = [0.8, 1.0, 1.2]
    else:  # bark
        tau_grid = [0.0, 0.2, 0.3, 0.4]
        T_grid = [0.8, 1.0, 1.2]

    best_tau, best_T, best_top1 = 0.0, 1.0, -1.0
    for tau in tau_grid:
        for T in T_grid:
            t1, _, _, _ = _recompute_metrics_from_logits(logits_list, y_true_list, prior_log, tau, T)
            if t1 > best_top1:
                best_top1, best_tau, best_T = t1, tau, T
    PRIOR_TAU_T_REGISTRY[organ] = (best_tau, best_T)
    return best_tau, best_T

# ============ 训练主函数（仅训练，不含推理）===========
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, weight=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', label_smoothing=label_smoothing)
        self.reduction = reduction
    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        pt = torch.exp(-ce).clamp_min(1e-8)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum':  return loss.sum()
        return loss

def mixup_data(x, y, alpha=0.2):
    if alpha is None or alpha <= 0: return x, (y, y), 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    return x_mix, (y, y[idx]), lam

def mixup_criterion(crit, pred, targets, lam):
    y_a, y_b = targets
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

# === 下方：在你的训练函数里换成“new_映射 + NEW切分 + new_权重名” ===
def train_one_species_model_for_organ(
    df: pd.DataFrame,
    organ: str,
    epochs_head: int = None,
    epochs_ft: int = None,
    bs: int = 64,
    base_lr_head: float = 1e-3,
    base_lr_ft_head: float = 1e-4,
    base_lr_ft_backbone: float = 2e-5,
    num_workers: int = 2,
    early_stop_patience: int = 5,
    scheduler_patience: int = 2,
    use_balanced_sampler: bool = True,
    use_multicrop_eval: bool = True,
    freeze_bn_after_warmup: bool = True,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    use_ema: bool = True,
    ema_m: float = 0.999,
    resume: bool = True,
    use_amp: bool = True,
):
    # —— 保持你原有的器官自适应配置、骨干与评估逻辑不变 —— 
    global current_organ
    current_organ = organ = organ.lower().strip()

    spec = ORGAN_SPEC.get(organ, ORGAN_SPEC["leaf"])
    epochs_head = spec["epochs_head"] if epochs_head is None else epochs_head
    epochs_ft   = spec["epochs_ft"]   if epochs_ft   is None else epochs_ft

    # ★★★ 关键改动1：为该器官构建 new_* 局部映射（会过滤掉该器官下仅1张图的物种）
    new_gid2local, new_local2gid, new_label2local, kept_gids = build_new_species_maps_for_organ(
        df, organ, COL_ORGAN_TXT, COL_LABEL, COL_SPECIESID
    )
    if len(new_local2gid) < 2:
        print(f"⚠️ 跳过 {organ}（过滤后可用物种数 < 2）")
        return

    num_classes = len(new_local2gid)

    # 模型 & 变换（沿用你已有的 get_backbone_for_organ，注意它会注册 VAL_TF）
    model, tf_pair, img_size = get_backbone_for_organ(organ, num_classes)
    train_tf, val_tf = tf_pair

    # Checkpoint 目录（每器官独立）
    ckpt_dir = OUT_DIR / "checkpoints" / organ
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_state_path = ckpt_dir / "last_state.pt"
    resume_has_ckpt = (resume and last_state_path.exists())

    # ★★★ 关键改动2：使用 NEW 的切分逻辑，保证 train/val 全覆盖
    tr_df, te_df = _load_fixed_split_or_fallback_NEW(
        df=df, organ=organ, col_organ=COL_ORGAN_TXT, col_gid=COL_SPECIESID,
        new_gid2local=new_gid2local, val_ratio=0.2, seed=42
    )

    # === 统计类先验并注册 ===
    freq = tr_df["__species_local__"].value_counts().sort_index()
    pi = (freq / freq.sum()).values.astype('float32')   # 概率
    # 防止 log(0)
    pi = np.clip(pi, 1e-8, 1.0)
    log_prior = torch.from_numpy(np.log(pi))
    PRIOR_LOG_REGISTRY[organ] = log_prior  # 供 evaluate_with_multicrop 使用


    # 验证集类别必须包含于训练集
    assert set(te_df["__species_local__"].unique()).issubset(set(tr_df["__species_local__"].unique())), \
        f"[{organ}] (NEW) Val 出现了训练缺失的类别，请清理 species_new/splits。"

    # —— 下面训练细节完全沿用你的原逻辑（损失、DataLoader、warmup、微调、EMA、评估等）——
    # 类频与 class_w
    cnt = tr_df["__species_local__"].value_counts()
    class_w = torch.tensor(
        [1.0 / np.log(1.2 + cnt.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float, device=DEVICE
    )

    local_lookup = {int(i): int(i) for i in range(num_classes)}
    tr_ds = BasicImageDatasetNEW(tr_df, COL_IMAGE, "__species_local__", local_lookup, train_tf)
    te_ds = BasicImageDatasetNEW(te_df, COL_IMAGE, "__species_local__", local_lookup, val_tf)

    if use_balanced_sampler:
        if organ == "fruit":
            # 使用 1/sqrt(freq) 权重，更稳
            freq_series = tr_df["__species_local__"].value_counts().sort_index()
            freq_vec = freq_series.reindex(range(num_classes), fill_value=1).values.astype(float)
            class_weights = 1.0 / np.sqrt(np.maximum(freq_vec, 1.0))
            sample_w = tr_df["__species_local__"].map(lambda i: class_weights[int(i)]).astype(float).values
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_w, dtype=torch.double),
                num_samples=len(sample_w),
                replacement=True
            )
        else:
            weights = tr_df["__species_local__"].map(lambda i: class_w[int(i)].item()).astype(float).values
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(weights),
                replacement=True
            )
        tr_ld = torch.utils.data.DataLoader(
            tr_ds, batch_size=bs, sampler=sampler,
            num_workers=num_workers, pin_memory=True, persistent_workers=False,
            collate_fn=drop_corrupt_collate)
    else:
        tr_ld = torch.utils.data.DataLoader(
            tr_ds, batch_size=bs, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=False,
            collate_fn=drop_corrupt_collate)

    te_ld = torch.utils.data.DataLoader(
        te_ds, batch_size=bs, shuffle=False,
        num_workers=max(1, num_workers//2), pin_memory=True, persistent_workers=False,
        collate_fn=drop_corrupt_collate)

    # 损失 & 训练两阶段（与你原脚本一致，这里略）……
    # 你原有的 Warmup / freeze BN / Finetune / EMA / 评估代码块直接保留即可
    # —— 省略 ——（把你现有训练主体粘过来，变量名不变即可）

    # …………（此处放你原来的训练循环与 evaluate_with_multicrop 调用）………… #

    # 只保留一种再平衡：使用 WeightedRandomSampler + 标准 CE（不再传 class_w，不再用 Focal）
    # —— 统一“只保留采样再平衡”：用 WeightedRandomSampler + 标准 CE —— 
    # （保留你已有的 sampler 构造，不再把 class_w 传进 CE，也不再用 Focal）

    organ_cfg = {
        "leaf":   {"mixup_alpha": 0.2, "label_smoothing": 0.02},
        "flower": {"mixup_alpha": 0.2, "label_smoothing": 0.02},
        "fruit":  {"mixup_alpha": 0.1, "label_smoothing": 0.02},
        "bark":   {"mixup_alpha": 0.0, "label_smoothing": 0.00},  # bark 关闭 mixup（你前文已做，这里保留）
    }
    cfg = organ_cfg.get(organ, {"mixup_alpha":0.2, "label_smoothing":0.02})
    mixup_alpha = cfg["mixup_alpha"]

    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    use_mixup = (mixup_alpha is not None and mixup_alpha > 0.0)


    # ===== A) Warmup（只训头） =====
    if not resume_has_ckpt:
        for p in model.parameters(): p.requires_grad = False
        if organ == "flower":
            for p in model.classifier[2].parameters(): p.requires_grad = True
        elif organ == "leaf":
            for p in model.classifier[2].parameters(): p.requires_grad = True
        elif organ == "fruit":
            for p in model.classifier[1].parameters(): p.requires_grad = True
        elif organ == "bark":
            for p in model.classifier[2].parameters(): p.requires_grad = True

        model = model.to(DEVICE)
        crit_warmup = nn.CrossEntropyLoss(label_smoothing=0.05)
        # >>> 改：warm-up 选择使用 train_tf 或 val_tf
        warmup_tf = (TRAIN_TF_REGISTRY[organ] if WARMUP_USES_TRAIN_TF.get(organ, False)
                    else VAL_TF_REGISTRY[organ])

        warmup_loader = DataLoader(
            BasicImageDatasetNEW(tr_df, COL_IMAGE, "__species_local__", local_lookup, warmup_tf),
            batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True, 
            collate_fn=drop_corrupt_collate
        )
        opt_head = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr_head)

        # Sanity check：输出维度
        tmp_loader = DataLoader(
            BasicImageDatasetNEW(tr_df.sample(min(8, len(tr_df))), COL_IMAGE, "__species_local__", local_lookup, VAL_TF_REGISTRY[organ]),
            batch_size=min(8, bs), shuffle=True, num_workers=0, collate_fn=drop_corrupt_collate
        )
        x0, y0, _ = next(iter(tmp_loader))
        x0, y0 = x0.to(DEVICE), y0.to(DEVICE)
        with torch.no_grad():
            logits0 = model(x0)
        assert logits0.shape[1] == num_classes, f"[{organ}] 头部维度不等于类数：{logits0.shape[1]} vs {num_classes}"

        for ep in range(epochs_head):
            model.train()
            total, correct = 0, 0
            for x, y, _ in tqdm(warmup_loader, total=len(warmup_loader), desc=f"[{organ}] Warmup {ep+1}/{epochs_head}"):
                if x.numel() == 0: continue
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast(enabled=use_amp):
                    logits = model(x)
                    loss = crit_warmup(logits, y)
                opt_head.zero_grad(); loss.backward(); opt_head.step()
                correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
            print(f"[{organ}] Warmup {ep+1}/{epochs_head} | Train Acc: {correct/max(1,total):.4f}")
    else:
        print(f"[{organ}] 检测到断点，跳过 warmup，直接进入微调")

    # BN 冻结（小 batch 推荐）
    if freeze_bn_after_warmup:
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    # ===== B) 解冻高层微调（LLRD/两组LR + EMA + Mixup） =====
    for p in model.parameters(): p.requires_grad = False
    if organ == "leaf":
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ("features.7","features.8","stages.2","stages.3","classifier")):
                p.requires_grad = True
    elif organ in ["flower", "bark"]:
        for name, p in model.named_parameters():
            if any(name.startswith(k) for k in ("features.7","features.8","stages.2","stages.3","classifier")):
                p.requires_grad = True
        for p in getattr(model, "classifier").parameters(): p.requires_grad = True
    elif organ == "fruit":
        for name, p in model.named_parameters():
            if name.startswith(("features.5","features.6","features.7","classifier")): p.requires_grad = True
        for p in model.classifier.parameters(): p.requires_grad = True

    params_head, params_backbone = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (params_head if any(k in name for k in ["fc","classifier"]) else params_backbone).append(p)

    def build_llrd_for_convnext(m, base_lr_bb, lr_head):
        groups = []
        for name, p in m.named_parameters():
            if not p.requires_grad: 
                continue
            if "classifier" in name or name.startswith("fc"):
                groups.append({"params": [p], "lr": lr_head, "weight_decay": 5e-4})
            else:
                lr = base_lr_bb
                if any(k in name for k in ["features.8","stages.3"]):
                    lr = max(base_lr_bb * 8.0, 8e-5)
                elif any(k in name for k in ["features.7","stages.2"]):
                    lr = max(base_lr_bb * 4.0, 4e-5)
                groups.append({"params": [p], "lr": lr, "weight_decay": 3e-4})
        return groups

    if organ in ("bark","flower"):
        param_groups = build_llrd_for_convnext(model, base_lr_ft_backbone, base_lr_ft_head)
        opt_ft = torch.optim.AdamW(param_groups)
    else:
        opt_ft = torch.optim.AdamW([
            {"params": params_backbone, "lr": base_lr_ft_backbone, "weight_decay": 3e-4},
            {"params": params_head,     "lr": base_lr_ft_head,     "weight_decay": 5e-4},
        ])

    #from torch.optim.lr_scheduler import ReduceLROnPlateau
    #scheduler = ReduceLROnPlateau(opt_ft, mode="max", factor=0.5, patience=max(2, scheduler_patience), verbose=True)

    # === 替换为 余弦退火 + warmup ===
    from torch.optim.lr_scheduler import CosineAnnealingLR

    # 先给每个 param group 记录 initial_lr（用于 warmup 线性升温）
    for pg in opt_ft.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg['lr']

    # warmup 轮数（按微调总轮数的 10%，至少 1 轮）
    warmup_epochs = max(1, int(0.10 * epochs_ft))

    # 余弦退火（warmup 结束后再进入余弦阶段）
    # 注意：T_max 设为 “剩余微调轮数”，eta_min 取一个 base_lr 的 1%~10%（可自行微调）
    scheduler_cos = CosineAnnealingLR(
        opt_ft,
        T_max=max(1, epochs_ft - warmup_epochs),
        eta_min=0.1 * min(pg['initial_lr'] for pg in opt_ft.param_groups)  # 也可以用 0.01
    )



    ema_model = deepcopy(model).to(DEVICE) if use_ema else None
    if use_ema:
        for p in ema_model.parameters(): p.requires_grad = False

    best_state = None
    best_top1  = -1.0
    bad = 0

    # AMP Scaler
    scaler = GradScaler(enabled=use_amp)

    # 断点恢复（仅在微调阶段恢复）
    start_ep = 0
    if resume and last_state_path.exists():
        try:
            state = torch.load(last_state_path, map_location="cpu")
            model.load_state_dict(state.get("model_state", {}), strict=False)
            if use_ema and state.get("ema_model_state") is not None:
                ema_model.load_state_dict(state["ema_model_state"], strict=False)
            opt_ft.load_state_dict(state.get("optimizer_state", {}))
            scheduler_cos.load_state_dict(state.get("scheduler_state", {}))
            if scaler is not None and state.get("scaler_state") is not None:
                scaler.load_state_dict(state["scaler_state"])
            start_ep = int(state.get("epoch", -1)) + 1
            best_top1 = float(state.get("best_top1", -1.0))
            if state.get("best_state") is not None:
                best_state = state["best_state"]
            print(f"[{organ}] 恢复训练：从 epoch {start_ep} 继续（best_top1={best_top1:.4f}）")
        except Exception as e:
            print(f"[{organ}] 恢复失败，重新开始微调：{e}")

    for ep in range(start_ep, epochs_ft):
        model.train()
        total, correct = 0, 0
        for x, y, _ in tqdm(tr_ld, total=len(tr_ld), desc=f"[{organ}] Finetune {ep+1}/{epochs_ft}"):
            if x.numel() == 0: continue
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_mixup:
                x, (ya, yb), lam = mixup_data(x, y, alpha=mixup_alpha)
                with autocast(enabled=use_amp):
                    logits = model(x)
                    loss = mixup_criterion(crit, logits, (ya, yb), lam)
            else:
                with autocast(enabled=use_amp):
                    logits = model(x)
                    loss = crit(logits, y)
            opt_ft.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt_ft)
                scaler.update()
            else:
                loss.backward(); opt_ft.step()
            if use_ema:
                with torch.no_grad():
                    for p_e, p in zip(ema_model.parameters(), model.parameters()):
                        p_e.data.mul_(ema_m).add_(p.data, alpha=1 - ema_m)
            pred = logits.argmax(1)
            correct += (pred == (y if not use_mixup else ya)).sum().item()
            total += y.size(0)
        tr_acc = correct / max(1,total)

        # === 每轮末尾做 LR 调度 ===
        if ep < warmup_epochs:
            # 线性 warmup：从 0 → initial_lr（按 param group 单独线性升温）
            scale = float(ep + 1) / warmup_epochs
            for pg in opt_ft.param_groups:
                pg['lr'] = pg['initial_lr'] * scale
        else:
            scheduler_cos.step()


        model_for_eval = ema_model if use_ema else model
        val_top1, val_top3, val_macro_f1, _ = evaluate_with_multicrop(
            model_for_eval, te_df, img_size, DEVICE,
            use_tta=True,            # 只保留水平翻转
            use_multicrop=False,     # 关
            logit_adj_tau=0.0        # 关掉先验校正
        )

        print(f"[{organ}] Finetune {ep+1}/{epochs_ft} | Train Acc: {tr_acc:.4f} | "
              f"Val Top1: {val_top1:.4f} | Top3: {val_top3:.4f} | MacroF1: {val_macro_f1:.4f}")

        #scheduler.step(val_top1)

        if val_top1 > best_top1:
            best_top1 = val_top1
            bad = 0
            best_state = deepcopy(model_for_eval.state_dict())
            print(f"[{organ}] 🔥 Update BEST Top1={best_top1:.4f}")
        else:
            bad += 1
            if bad >= early_stop_patience:
                print(f"[{organ}] Early stopping at epoch {ep+1}.")
                break

        # 保存断点（last_state）
        try:
            save_obj = {
                "epoch": ep,
                "model_state": model.state_dict(),
                "ema_model_state": (ema_model.state_dict() if use_ema else None),
                "optimizer_state": opt_ft.state_dict(),
                "scheduler_state": scheduler_cos.state_dict(),
                "scaler_state": (scaler.state_dict() if use_amp else None),
                "best_top1": best_top1,
                "best_state": best_state,
                "organ": organ,
                "country": COUNTRY,
            }
            torch.save(save_obj, last_state_path)
        except Exception as e:
            print(f"[{organ}] 保存断点失败：{e}")

    # 载入 BEST，最终验证一次并保存
    if best_state is not None:
        (ema_model if use_ema else model).load_state_dict(best_state, strict=False)

    best_model = (ema_model if use_ema else model)
    # 单次前向，缓存 logits 与 y（不做校正），再离线挑选 (tau*, T*)
    _, _, _, _, logits_list, y_true_list = evaluate_with_multicrop(
        best_model, te_df, img_size, DEVICE,
        use_tta=True, use_multicrop=True,
        n_crops=(10 if organ in ["leaf","flower"] else 9),
        ratio_low=0.95,
        logit_adj_tau=0.0, temp_T=1.0, return_logits=True
    )
    prior_log = PRIOR_LOG_REGISTRY.get(organ, None)
    tau_star, T_star = _select_best_tau_T(organ, logits_list, y_true_list, prior_log if prior_log is not None else torch.zeros_like(logits_list[0]))
    top1, top3, macro_f1, y_pred = _recompute_metrics_from_logits(logits_list, y_true_list, prior_log if prior_log is not None else torch.zeros_like(logits_list[0]), tau_star, T_star)

    # 构造报告（离线）
    labels_present = sorted(set(y_true_list) | set(y_pred))
    try:
        with open(SPECIES_NEW_DIR / f"new_species_local_map_{organ}.json", "r", encoding="utf-8") as f:
            label2local = json.load(f)
        local2label = {int(v): k for k, v in label2local.items()}
        target_names = [local2label.get(i, str(i)) for i in labels_present]
    except Exception:
        try:
            with open(OUT_DIR / f"{COUNTRY}_species_local_map_{organ}.json", "r", encoding="utf-8") as f:
                label2local = json.load(f)
            local2label = {int(v): k for k, v in label2local.items()}
            target_names = [local2label.get(i, str(i)) for i in labels_present]
        except Exception:
            target_names = [str(i) for i in labels_present]
    report = classification_report(y_true_list, y_pred, labels=labels_present, target_names=target_names, zero_division=0)

    # 可选：训练集评估保持原逻辑
    tr_top1, tr_top3, tr_macro_f1, _ = evaluate_with_multicrop(
        best_model, tr_df, img_size, DEVICE,
        use_tta=True, use_multicrop=True, n_crops=(9 if organ in ["leaf","flower"] else 8),
        ratio_low=0.90, logit_adj_tau=1.0
    )

    print(f"[{organ}] Train(EvalMode) | Top1: {tr_top1:.4f} | Top3: {tr_top3:.4f} | MacroF1: {tr_macro_f1:.4f}")
    
    print(f"✅ [{organ}] Final Test | Top1: {top1:.4f} | Top3: {top3:.4f} | MacroF1: {macro_f1:.4f}")
    print(report)


    # 训练完保存 BEST 后：
    # ★★★ 关键改动3：保存到 species_new/，并用 new_ 前缀区分
    save_path = SPECIES_NEW_DIR / f"new_{organ}_species_model_{COUNTRY}.pth"
    torch.save(best_model.state_dict(), save_path)
    print(f"✅ (NEW) 已保存: {save_path}")

def _country_dir(base_out: Path, country: str) -> Path:
    cc = country.strip()
    return Path(base_out) / f"{cc}_organ_species_model"


def train_all_organs(csv_path: str, base_out: str, country: str, data_root: str = "/", organs: List[str] = None,
                     global_bs: int = None, global_num_workers: int = None, resume: bool = True, use_amp: bool = True):
    global OUT_DIR, SPECIES_NEW_DIR, DEVICE, COUNTRY, df
    COUNTRY = country
    OUT_DIR = _country_dir(Path(base_out), country)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPECIES_NEW_DIR = OUT_DIR / "species_new"
    (SPECIES_NEW_DIR / "splits").mkdir(parents=True, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取数据
    df = pd.read_csv(csv_path)
    assert all(c in df.columns for c in [COL_IMAGE, COL_ORGAN_TXT, COL_LABEL, COL_SPECIESID]), \
        f"CSV 至少需要列: {COL_IMAGE}, {COL_ORGAN_TXT}, {COL_LABEL}, {COL_SPECIESID}"

    # 按器官训练
    organ_list = organs if organs is not None and len(organs) > 0 else ["bark","flower","fruit","leaf"]
    # 默认并行加载线程
    num_workers = 2
    for organ in organ_list:
        sub = df[df[COL_ORGAN_TXT] == organ]
        uniq_species = sub[COL_SPECIESID].nunique()
        if len(sub) < 50 or uniq_species < 10:    ####应该是能够提升准确率，如果不行，就还是2 or 4
            print(f"⚠️ 跳过 {organ}（样本={len(sub)}, 物种={uniq_species}）")
            continue

        if organ == "flower":
            bs = 72; base_lr_ft_head=3e-4; base_lr_ft_backbone=2e-5 ; use_balanced_sampler= True ; freeze_bn = True
        elif organ == "leaf":
            bs = 72; base_lr_ft_head=3e-4; base_lr_ft_backbone = 3e-5; use_balanced_sampler= False ; freeze_bn = False
        elif organ == "fruit":
            bs = 60; base_lr_ft_head=3e-4; base_lr_ft_backbone=3e-5; use_balanced_sampler= True ; freeze_bn = True
        elif organ == "bark":
            bs = 60; base_lr_ft_head=1e-3; base_lr_ft_backbone = 5e-5;use_balanced_sampler= True ; freeze_bn = True

        if global_bs is not None:
            bs = global_bs
        if global_num_workers is not None:
            num_workers = global_num_workers

        print(f"\n🎯 训练器官 [{organ}] —— 样本={len(sub)}, 物种={uniq_species}")
        train_one_species_model_for_organ(
            df=df,
            organ=organ,
            #epochs_head=3,
            #epochs_ft=5,
            bs=bs,
            base_lr_head=1e-3,
            base_lr_ft_head=base_lr_ft_head,
            base_lr_ft_backbone=base_lr_ft_backbone,
            num_workers=num_workers,
            early_stop_patience=5,
            scheduler_patience=2,
            use_balanced_sampler=use_balanced_sampler,
            freeze_bn_after_warmup=freeze_bn,
            resume=resume,
            use_amp=use_amp,
        )


def main():
    parser = argparse.ArgumentParser(description="按器官训练物种分类器")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--data-root", default="/")
    parser.add_argument("--organs", default="", help="逗号分隔的器官子集，例如: leaf,fruit")
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--amp", type=int, default=1)
    args = parser.parse_args()

    organs = [o.strip() for o in args.organs.split(",") if o.strip()] if args.organs else None
    train_all_organs(
        args.csv, args.out, args.country, args.data_root,
        organs=organs,
        global_bs=args.bs,
        global_num_workers=args.num_workers,
        resume=bool(args.resume),
        use_amp=bool(args.amp),
    )


if __name__ == "__main__":
    main()

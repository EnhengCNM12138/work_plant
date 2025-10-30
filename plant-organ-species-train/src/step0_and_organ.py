import json
import argparse
from pathlib import Path
import pandas as pd

# === 列名定义（统一规范）===
COL_IMAGE     = "image_path"   # 图片路径
COL_ORGAN_TXT = "organ"        # 器官文字：leaf/flower/fruit/bark
COL_ORGAN_ID  = "organ_id"     # 器官数值：如 0/1/2/3（或 1/2/3/4）
COL_LABEL     = "label"        # 植物拉丁学名（文本）
COL_SPECIESID = "species"      # 植物全局 ID（数值，label 的数值化）


def _country_dir(base_out: Path, country: str) -> Path:
    cc = country.strip()
    return Path(base_out) / f"{cc}_organ_species_model"


def run_step0(csv_path: str, base_out: str, country: str):
    out_dir = _country_dir(Path(base_out), country)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    assert all(c in df.columns for c in [COL_IMAGE, COL_ORGAN_TXT, COL_ORGAN_ID, COL_LABEL, COL_SPECIESID]), \
        f"CSV 至少需要列: {COL_IMAGE}, {COL_ORGAN_TXT}, {COL_ORGAN_ID}, {COL_LABEL}, {COL_SPECIESID}"

    df[COL_ORGAN_TXT] = df[COL_ORGAN_TXT].astype(str).str.strip().str.lower()
    df[COL_LABEL]     = df[COL_LABEL].astype(str).str.strip()

    # 1) <CC>_organ_classes.json（基于 organ↔organ_id 一致性）
    pairs = df[[COL_ORGAN_TXT, COL_ORGAN_ID]].drop_duplicates()
    cnt1 = pairs.groupby(COL_ORGAN_TXT)[COL_ORGAN_ID].nunique()
    cnt2 = pairs.groupby(COL_ORGAN_ID)[COL_ORGAN_TXT].nunique()
    if (cnt1 > 1).any() or (cnt2 > 1).any():
        raise ValueError("检测到 organ 与 organ_id 不是一一对应，请检查数据。")

    organ_classes = dict(pairs.sort_values(COL_ORGAN_ID).values)  # 文本→id
    with open(out_dir / f"{country}_organ_classes.json", "w", encoding="utf-8") as f:
        json.dump(organ_classes, f, ensure_ascii=False, indent=2)
    print("✅ organ_classes.json 已保存：", out_dir / f"{country}_organ_classes.json")

    # 2) 每器官生成局部映射（训练 head 用），文件名带 <CC> 前缀
    for organ_txt in organ_classes.keys():
        sub = df[df[COL_ORGAN_TXT] == organ_txt].copy()
        uniq = sub[[COL_LABEL, COL_SPECIESID]].drop_duplicates()
        uniq = uniq.sort_values([COL_SPECIESID, COL_LABEL]).reset_index(drop=True)
        uniq["local_id"] = range(len(uniq))

        label2local = {row[COL_LABEL]: int(row["local_id"]) for _, row in uniq.iterrows()}
        gid2local   = {str(int(row[COL_SPECIESID])): int(row["local_id"]) for _, row in uniq.iterrows()}
        local2gid   = {str(int(row["local_id"])): int(row[COL_SPECIESID]) for _, row in uniq.iterrows()}

        with open(out_dir / f"{country}_species_local_map_{organ_txt}.json", "w", encoding="utf-8") as f:
            json.dump(label2local, f, ensure_ascii=False, indent=2)
        with open(out_dir / f"{country}_species_global2local_{organ_txt}.json", "w", encoding="utf-8") as f:
            json.dump(gid2local, f, ensure_ascii=False, indent=2)
        with open(out_dir / f"{country}_species_local2global_{organ_txt}.json", "w", encoding="utf-8") as f:
            json.dump(local2gid, f, ensure_ascii=False, indent=2)

        print(f"✅ [{organ_txt}] 局部映射已保存：labels={len(label2local)} -> {out_dir}")
# =======================
# Step 1: 依赖 & Dataset
# =======================
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
    ResNet18_Weights,
    ResNet50_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device =", DEVICE)

class BasicImageDataset(Dataset):
    """
    通用图像数据集：通过 label_lookup 决定监督标签的来源（organ 或 species local_id）
    - label_col 可是 organ_id 或 "__species_local__"
    - label_lookup 把【真实值】映射到【训练用的连续id】，这里通常是恒等映射
    """
    def __init__(self, df: pd.DataFrame, image_col: str, label_col: str,
                 label_lookup: Dict[int, int], tfm: transforms.Compose):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.label_lookup = label_lookup
        self.tfm = tfm

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.image_col]).convert("RGB")
        img = self.tfm(img)
        raw_label = int(row[self.label_col])
        label = self.label_lookup[raw_label]
        return img, label
from torchvision.transforms import AutoAugment, AutoAugmentPolicy, RandomErasing

def make_species_transforms(organ: str, img_size: int):
    organ = organ.lower().strip()

    # 骨架：val 始终 CenterCrop；train 使用 RandomResizedCrop + 适度颜色/几何扰动
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    if organ == "flower":
        train_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.15)),
            transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.80, 1.20)),
            transforms.RandomHorizontalFlip(p=0.5),
            AutoAugment(AutoAugmentPolicy.IMAGENET),          # 提升泛化（花色/形状）
            transforms.ColorJitter(0.20, 0.20, 0.12, 0.04),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            RandomErasing(p=0.20, scale=(0.02, 0.10), value='random'),
        ])

    elif organ == "leaf":
        train_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.10)),
            transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.18, 0.18, 0.10, 0.03),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
            RandomErasing(p=0.25, scale=(0.02, 0.12), value='random'),  # 叶脉鲁棒
        ])

    elif organ == "fruit":
        train_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.10)),
            transforms.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),            # ✅ 果实形变/朝向
            transforms.ColorJitter(0.15, 0.15, 0.10, 0.03),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])

    else:  # bark
        train_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.10)),
            transforms.RandomResizedCrop(img_size, scale=(0.60, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.10, 0.10, 0.06, 0.02),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])

    # ======= bark 训练增强（更接近评估视野，抑制形变） =======


    return train_tf, val_tf
# =======================
# Step 2: 器官分类器
# =======================

# ================ Cell 1: Imports & Utils ================
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 更快


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ================ Cell 2: Config ================
@dataclass
class TrainConfig:
    csv: str = "/mnt/e/code/plants-classification-conda/real_data/plant_China.csv"  # 改成你的 CSV 路径
    data_root: str = "/"                                 # 若 image_path 为绝对路径，保持 "/" 即可
    out: str = "./real_data"          # 输出目录

    backbone: str = 'convnext_small'                        # timm 任意骨干
    pretrained: bool = True
    #img_size: int = 256 原设置为256
    img_size: int = 384
    batch_size: int = 64
    epochs: int = 100
    lr: float = 2e-4
    wd: float = 5e-2
    warmup_epochs: float = 3.0
    num_workers: int = 8
    amp: bool = True
    grad_accum: int = 1
    aux_weight: float = 0.35
    label_smoothing: float = 0.05
    seed: int = 42
    train_split: float = 0.9        # 训练集比例，分层划分
    patience: int = 20              # 早停
    save_every: int = 0             # >0 表示每 N 轮额外保存一次



    # 采样配置（缓解类不平衡）
    use_weighted_sampler: bool = True
    sampler_power: float = 0.7      # 1.0=完全按 1/freq；<1 软化

    # 温度标定（验证集）
    do_temperature_scaling: bool = True

    # 推理用阈值（训练不使用，仅保留）
    confidence_tau: float = 0.55


CFG = TrainConfig()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================ Cell 3: Dataset ================
ORGANS_STD = ["bark", "flower", "fruit", "leaf"]
ORGAN2ID = {k: i for i, k in enumerate(ORGANS_STD)}

# 常见同义归一，可按需扩展
ORGAN_NORMALIZE = {
    'flower': 'flower', 'flw': 'flower', 'flo': 'flower', 'flor': 'flower',
    'leaf': 'leaf', 'leaves': 'leaf', 'foliage': 'leaf',
    'fruit': 'fruit', 'frt': 'fruit', 'seedpod': 'fruit',
    'bark': 'bark', 'trunk': 'bark', 'stem': 'bark', 'branch': 'bark'
}


def normalize_organ(name: str) -> str:
    if name is None:
        return None
    key = str(name).strip().lower()
    return ORGAN_NORMALIZE.get(key, key)


class PlantOrganDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str, img_size: int, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.img_size = img_size
        self.train = train

        '''if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
    
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.05)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])'''

        if train:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.10)),
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.85, 1.15)),  # ↑ 提高下限，保细节
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.RandomPerspective(distortion_scale=0.2)
                ], p=0.2),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
    
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.05)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['image_path']
        if not os.path.isabs(path):
            path = os.path.join(self.data_root, path)
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # 若图片损坏，使用全黑占位，保证训练不中断
            img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        x = self.tf(img)
        y = int(row['label_id'])
        return x, y


# ================ Cell 4: Model ================
class OrgansHier(nn.Module):
    """共享骨干 + gate/head，输出 4 类概率（软组合）。"""
    def __init__(self, backbone: str = 'convnext_tiny', pretrained: bool = True, drop_path_rate: float = 0.3):
        super().__init__()
        # 可调 drop_path 有助于泛化
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        feat_dim = self.backbone.num_features
        self.gate = nn.Linear(feat_dim, 2)     # 0: FL (flower+leaf), 1: FB (fruit+bark)
        self.head_fl = nn.Linear(feat_dim, 2)  # [flower, leaf]
        self.head_fb = nn.Linear(feat_dim, 2)  # [fruit, bark]

    def forward(
        self, x, targets=None, aux_weight: float = 0.3, label_smoothing: float = 0.0, temperature: float = 1.0,
        w_main: torch.Tensor | None = None,
        w_gate: torch.Tensor | None = None,
        w_fl_head: torch.Tensor | None = None,
        w_fb_head: torch.Tensor | None = None,
        gate_margin_weight: float = 0.08, gate_hi: float = 0.7, gate_lo: float = 0.3
    ):
        feats = self.backbone(x)
        gate_logits = self.gate(feats) / temperature
        fl_logits = self.head_fl(feats) / temperature
        fb_logits = self.head_fb(feats) / temperature

        g = F.softmax(gate_logits, dim=1)[:, 0]  # P(FL)
        p_fl = F.softmax(fl_logits, dim=1)       # [flower, leaf]
        p_fb = F.softmax(fb_logits, dim=1)       # [fruit, bark]

        P = torch.stack([
            g * p_fl[:, 0],          # flower
            g * p_fl[:, 1],          # leaf
            (1 - g) * p_fb[:, 0],    # fruit
            (1 - g) * p_fb[:, 1],    # bark
        ], dim=1).clamp_min(1e-8)

        outputs = {
            'probs': P,
            'gate': g,
            'gate_logits': gate_logits,
            'fl_logits': fl_logits,
            'fb_logits': fb_logits,
        }

        if targets is not None:
            # 主损失（四类）— 带类权重
            loss_main = nll_loss_with_label_smoothing(P.log(), targets, smoothing=label_smoothing, weight=w_main)

            # gate 损失（FL/FB）— 带权重
            is_FL = (targets == 0) | (targets == 1)
            gate_t = torch.where(is_FL, torch.zeros_like(targets), torch.ones_like(targets))  # 0:FL,1:FB
            loss_gate = F.cross_entropy(gate_logits, gate_t, weight=w_gate)

            # 子头损失（各自两类）— 带权重
            fl_mask = is_FL
            fb_mask = ~is_FL
            loss_fl = (F.cross_entropy(fl_logits[fl_mask], (targets[fl_mask] % 2), weight=w_fl_head)
                       if fl_mask.any() else torch.tensor(0., device=x.device))
            loss_fb = (F.cross_entropy(fb_logits[fb_mask], ((targets[fb_mask] - 2) % 2), weight=w_fb_head)
                       if fb_mask.any() else torch.tensor(0., device=x.device))

            total_loss = loss_main + aux_weight * (loss_gate + loss_fl + loss_fb)

            # gate margin 正则：让 FL 样本 g 更大、FB 样本 g 更小（轻量）
            if gate_margin_weight > 0:
                loss_gate_margin = 0.
                if fl_mask.any():
                    loss_gate_margin = loss_gate_margin + F.relu(gate_hi - g[fl_mask]).mean()
                if fb_mask.any():
                    loss_gate_margin = loss_gate_margin + F.relu(g[fb_mask] - gate_lo).mean()
                total_loss = total_loss + gate_margin_weight * loss_gate_margin

            outputs['loss'] = total_loss

        return outputs


def nll_loss_with_label_smoothing(
    log_probs: torch.Tensor, targets: torch.Tensor, smoothing: float = 0.0, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """
    log_probs: [B, C] 的对数概率
    weight:   [C] 的类权重（可为 None）
    """
    if smoothing <= 0:
        return F.nll_loss(log_probs, targets, weight=weight)

    n_classes = log_probs.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)

    per_class_loss = -true_dist * log_probs  # [B,C]
    if weight is not None:
        per_class_loss = per_class_loss * weight.view(1, -1)
    return per_class_loss.sum(dim=1).mean()



# ================ Cell 5: Data Loading Helpers ================

def load_and_split(csv_path: str, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    # 归一化器官
    df['organ_norm'] = df['organ'].apply(normalize_organ)
    df = df[df['organ_norm'].isin(ORGANS_STD)].copy()
    df['label_id'] = df['organ_norm'].map(ORGAN2ID)

    # 分层划分
    train_df, val_df = train_test_split(
        df, test_size=1.0 - train_ratio, random_state=seed, stratify=df['label_id']
    )
    return train_df, val_df


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader, Dict]:
    train_df, val_df = load_and_split(cfg.csv, cfg.train_split, cfg.seed)

    stats = {
        'train_counts': train_df['label_id'].value_counts().to_dict(),
        'val_counts': val_df['label_id'].value_counts().to_dict(),
    }

    train_ds = PlantOrganDataset(train_df, cfg.data_root, cfg.img_size, train=True)
    val_ds = PlantOrganDataset(val_df, cfg.data_root, cfg.img_size, train=False)

    if cfg.use_weighted_sampler:
        counts = train_df['label_id'].value_counts().sort_index().values.astype(float)
        inv = (1.0 / (counts + 1e-6)) ** cfg.sampler_power
        class_weights = inv / inv.sum() * len(counts)
        sample_weights = train_df['label_id'].map(lambda c: class_weights[c]).values
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size*2, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, stats


# ================ Cell 6: Optimizer & Scheduler ================

def build_optimizer(model: nn.Module, cfg: TrainConfig):
    # norm/bias 不做 weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndimension() == 1 or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    params = [
        {'params': decay, 'weight_decay': cfg.wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    return opt


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ================ Cell 7: Train & Eval ================

def train_one_epoch(
    model, loader, optimizer, scaler, device, cfg: TrainConfig, ema=None,
    w_main=None, w_gate=None, w_fl_head=None, w_fb_head=None,
    gate_margin_weight: float = 0.05, gate_hi: float = 0.6, gate_lo: float = 0.4
):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc='train', leave=False)
    optimizer.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(pbar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=cfg.amp):
            out = model(
                x, targets=y,
                aux_weight=cfg.aux_weight,
                label_smoothing=cfg.label_smoothing,
                w_main=w_main, w_gate=w_gate, w_fl_head=w_fl_head, w_fb_head=w_fb_head,
                gate_margin_weight=gate_margin_weight, gate_hi=gate_hi, gate_lo=gate_lo
            )
            loss = out['loss'] / cfg.grad_accum

        if cfg.amp:
            scaler.scale(loss).backward()
    
            loss.backward()

        if (i + 1) % cfg.grad_accum == 0:
            if cfg.amp:
                scaler.step(optimizer)
                scaler.update()
        
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if ema is not None:
                ema.update(model)

        total_loss += float(loss.item()) * x.size(0) * cfg.grad_accum
        n += x.size(0)
        pbar.set_postfix(loss=f"{total_loss / max(1, n):.4f}")

    return total_loss / max(1, n)


def evaluate(model, loader, device, cfg: TrainConfig) -> Dict:
    model.eval()
    all_probs, all_targets, all_logits = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # TTA: 原图
            out1 = model(x, targets=None)
            probs1 = out1['probs']

            # TTA: 水平翻转
            x_flip = torch.flip(x, dims=[3])
            out2 = model(x_flip, targets=None)
            probs2 = out2['probs']

            probs = ((probs1 + probs2) * 0.5).clamp_min(1e-8)
            all_probs.append(probs.detach().cpu())
            all_targets.append(y.detach().cpu())
            all_logits.append(probs.log().detach().cpu())  # “伪 logits”供温度标定

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()

    preds = probs.argmax(axis=1)
    acc = (preds == targets).mean()
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average='macro')
    cm = confusion_matrix(targets, preds, labels=[0,1,2,3])
    rep = classification_report(targets, preds, target_names=ORGANS_STD, digits=4)

    return {
        'acc': float(acc),
        'bal_acc': float(bal_acc),
        'macro_f1': float(macro_f1),
        'cm': cm.tolist(),
        'report': rep,
        'probs': probs,
        'targets': targets,
        'logits': logits,
    }


def make_weights(counts, epoch, total_epochs, cap_ratio=2.0, alpha_final=0.8, warmup_ep=5):
    # 1) 退火：前 warmup_ep 轮从 0 -> alpha_final 线性增长
    if epoch <= warmup_ep:
        alpha = alpha_final * (epoch / max(1, warmup_ep))

        alpha = alpha_final
    inv = 1.0 / (counts + 1e-6)
    w = (inv ** alpha)
    # 2) 限幅：少数类/多数类的比值不超过 cap_ratio（例如 2 倍）
    w = w / w.min()
    w = np.clip(w, 1.0, cap_ratio)
    # 3) 归一到均值=1，数值更稳
    w = w / w.mean()
    # gate 的 FL/FB 权重也限幅
    n_FL = counts[0] + counts[1]
    n_FB = counts[2] + counts[3]
    wg = np.array([1.0/max(n_FL,1e-6), 1.0/max(n_FB,1e-6)], dtype=np.float32)
    wg = wg / wg.min()
    wg = np.clip(wg, 1.0, cap_ratio)
    wg = wg / wg.mean()
    return w, wg



# ================ Cell 8: Temperature Scaling (optional) ================
class _TempScalingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-3)


def fit_temperature(logits: np.ndarray, targets: np.ndarray, max_iter: int = 200, device: str = 'cuda') -> float:
    """在验证集 logits 上拟合标量温度，最小化 CE。"""
    model = _TempScalingModule().to(device)
    opt = torch.optim.LBFGS(model.parameters(), lr=0.01, max_iter=100)

    x = torch.from_numpy(logits).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(targets).to(device=device, dtype=torch.long)

    def _closure():
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        return loss

    for _ in range(max_iter):
        opt.step(_closure)
    T = float(model.temperature.detach().cpu().item())
    return T


# ================ Cell 9: Build Everything & Train (save-best) ================
import os, time, copy
import numpy as np
import torch


def train_organ_classifier(csv_path: str, base_out: str, country: str, data_root: str = "/"):
    global CFG
    # 初始化配置
    CFG = TrainConfig()
    CFG.csv = csv_path
    CFG.data_root = data_root
    out_dir = _country_dir(Path(base_out), country)
    out_dir.mkdir(parents=True, exist_ok=True)
    CFG.out = str(out_dir)
    set_seed(CFG.seed)
    ensure_dir(CFG.out)

    train_loader, val_loader, stats = build_dataloaders(CFG)
    print("Class counts (train):", stats['train_counts'])
    print("Class counts (val): ", stats['val_counts'])

    # 准备 counts（np.array），供 make_weights 使用
    counts_np = np.array([stats['train_counts'].get(i, 1) for i in range(4)], dtype=np.float32)

    model = OrgansHier(CFG.backbone, CFG.pretrained, drop_path_rate=0.2).to(DEVICE)
    optimizer = build_optimizer(model, CFG)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, CFG, steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.amp)
    ema = ModelEmaV2(model, decay=0.9999)

    # 保存目录与文件名（带国家缩写）
    save_dir = CFG.out
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, f"organ_classification_{country}_mybest.pth")
    last_path = os.path.join(save_dir, f"organ_classification_{country}_last.pth")

    best_metric = -1.0
    best_epoch  = -1
    best_state_dict = None
    no_improve = 0

    def _to_head_weights(w4):
        w_fl = np.array([w4[0], w4[1]], dtype=np.float32)
        w_fb = np.array([w4[2], w4[3]], dtype=np.float32)
        w_fl = w_fl / max(w_fl.mean(), 1e-6)
        w_fb = w_fb / max(w_fb.mean(), 1e-6)
        return w_fl, w_fb

    for epoch in range(1, CFG.epochs + 1):
        t0 = time.time()

        # —— 每个 epoch 计算当轮的退火权重（主/子头/gate）——
        w_main_np, w_gate_np = make_weights(
            epoch=epoch,
            total_epochs=CFG.epochs,
            counts=counts_np,
            cap_ratio=1.8,
            alpha_final=0.9,
            warmup_ep=6
        )
        w_fl_np, w_fb_np = _to_head_weights(w_main_np)

        # 转成 GPU tensor
        w_main    = torch.tensor(w_main_np, device=DEVICE, dtype=torch.float32)
        w_gate    = torch.tensor(w_gate_np, device=DEVICE, dtype=torch.float32)
        w_fl_head = torch.tensor(w_fl_np,  device=DEVICE, dtype=torch.float32)
        w_fb_head = torch.tensor(w_fb_np,  device=DEVICE, dtype=torch.float32)

        # gate margin 退火
        base_gate_margin = 0.08
        gm_w = base_gate_margin * min(1.0, epoch / 10.0)
        gate_hi, gate_lo = 0.7, 0.3

        # —— 训练一个 epoch —— 
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE, CFG, ema=ema,
            w_main=w_main, w_gate=w_gate, w_fl_head=w_fl_head, w_fb_head=w_fb_head,
            gate_margin_weight=gm_w, gate_hi=gate_hi, gate_lo=gate_lo
        )

        scheduler.step()

        # 用 EMA 评估（带 TTA）
        val_metrics = evaluate(ema.module, val_loader, DEVICE, CFG)
        t1 = time.time()
        print(f"Epoch {epoch:03d}: loss={train_loss:.4f} acc={val_metrics['acc']:.4f} "
              f"bal_acc={val_metrics['bal_acc']:.4f} macro_f1={val_metrics['macro_f1']:.4f} time={t1-t0:.1f}s")

        # 组合评价分（示例：macro_f1 权重更高）
        score = val_metrics['macro_f1'] * 1000 + val_metrics['bal_acc']

        # ======= ★ 出现更优就立刻保存 BEST ★ =======
        if score > best_metric:
            best_metric = score
            best_epoch  = epoch
            no_improve = 0

            # 深拷贝一份稳定的 EMA 权重
            best_state_dict = copy.deepcopy(ema.module.state_dict())

            # 先保存一个“临时无温度标定”的 best（便于断点观察/复现）
            pack = {
                "arch": "OrgansHier",
                "backbone": CFG.backbone,
                "drop_path_rate": 0.3,
                "state_dict": best_state_dict,
                "organs": ORGANS_STD,
                "organ2id": ORGAN2ID,
                "cfg": CFG.__dict__,
                "temperature": 1.0,
                "best_epoch": best_epoch,
                "best_metric": float(best_metric),
                "val_snapshot": {
                    "acc": float(val_metrics['acc']),
                    "bal_acc": float(val_metrics['bal_acc']),
                    "macro_f1": float(val_metrics['macro_f1']),
                },
            }
            torch.save(pack, best_path)
            print(f"保存 BEST: {best_path}")

        else:
            no_improve += 1

        if no_improve >= CFG.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {CFG.patience} epochs)")
            break

    # ============ 训练结束：保存“最后一次”的权重 ============
    torch.save({
        "arch": "OrgansHier",
        "backbone": CFG.backbone,
        "drop_path_rate": 0.3,
        "state_dict": ema.module.state_dict(),
        "organs": ORGANS_STD,
        "organ2id": ORGAN2ID,
        "cfg": CFG.__dict__,
        "temperature": 1.0,
        "last_epoch": epoch,
    }, last_path)
    print(f"已保存最后一次权重到: {last_path}")

    # ============ 用“最佳权重”做最终验证 & 温度标定，并覆盖保存 BEST ============
    if best_state_dict is None:
        best_state_dict = ema.module.state_dict()
        best_epoch = epoch
        best_metric = -1

    best_model = OrgansHier(CFG.backbone, CFG.pretrained, drop_path_rate=0.2).to(DEVICE)
    best_model.load_state_dict(best_state_dict, strict=True)
    best_model.eval()

    final_eval = evaluate(best_model, val_loader, DEVICE, CFG)

    T = 1.0
    if CFG.do_temperature_scaling:
        T = fit_temperature(final_eval['logits'], final_eval['targets'], device=DEVICE)
        print(f"Fitted temperature (BEST @ epoch {best_epoch}): {T:.3f}")

    final_pack = {
        "arch": "OrgansHier",
        "backbone": CFG.backbone,
        "drop_path_rate": 0.3,
        "state_dict": best_state_dict,
        "organs": ORGANS_STD,
        "organ2id": ORGAN2ID,
        "cfg": CFG.__dict__,
        "temperature": float(T),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
    }
    torch.save(final_pack, best_path)
    print(f"✅ Saved BEST organ classifier to: {best_path}")


def main():
    parser = argparse.ArgumentParser(description="Step0 和 器官分类器训练")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("step0", help="生成 organ/species 映射 JSON")
    p0.add_argument("--csv", required=True)
    p0.add_argument("--out", required=True)
    p0.add_argument("--country", required=True)

    p1 = sub.add_parser("train-organ", help="训练器官分类器")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--country", required=True)
    p1.add_argument("--data-root", default="/")

    args = parser.parse_args()

    if args.cmd == "step0":
        run_step0(args.csv, args.out, args.country)
    elif args.cmd == "train-organ":
        train_organ_classifier(args.csv, args.out, args.country, args.data_root)


if __name__ == "__main__":
    main()

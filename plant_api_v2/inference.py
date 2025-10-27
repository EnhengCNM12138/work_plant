import torch
import open_clip
import numpy as np
from PIL import Image
import collections
import requests
from io import BytesIO
import os
import uuid
import torch.nn.functional as F
import json
from pathlib import Path
from typing import List, Dict, Union, Tuple
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
)
import timm

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型路径配置
MODEL_DIR = Path("./weights")
OUT_DIR = MODEL_DIR

# 全局变量
clip_model = None
preprocess = None
organ_model = None
organ_transform = None
id2organ = None
species_models = {}
species_transforms = {}
species_id2label = {}

# 器官分类器模型类（来自organ_test.ipynb）
class OrgansHier(nn.Module):
    def __init__(self, backbone: str = 'convnext_small', pretrained: bool = True, drop_path_rate: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        feat_dim = self.backbone.num_features
        self.gate = nn.Linear(feat_dim, 2)     # 0: FL (flower+leaf), 1: FB (fruit+bark)
        self.head_fl = nn.Linear(feat_dim, 2)  # [flower, leaf]
        self.head_fb = nn.Linear(feat_dim, 2)  # [fruit, bark]

    def forward(self, x, temperature: float = 1.0):
        feats = self.backbone(x)
        gate_logits = self.gate(feats) / temperature
        fl_logits   = self.head_fl(feats) / temperature
        fb_logits   = self.head_fb(feats) / temperature
        g    = F.softmax(gate_logits, dim=1)[:, 0]  # P(FL)
        p_fl = F.softmax(fl_logits,   dim=1)        # [flower, leaf]
        p_fb = F.softmax(fb_logits,   dim=1)        # [fruit, bark]
        P = torch.stack([
            g * p_fl[:, 0],          # flower
            g * p_fl[:, 1],          # leaf
            (1 - g) * p_fb[:, 0],    # fruit
            (1 - g) * p_fb[:, 1],    # bark
        ], dim=1).clamp_min(1e-8)    # [B,4]
        return P

def build_eval_tf(organ: str, img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.05)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

def get_infer_backbone_for_organ(organ: str, num_classes: int) -> Tuple[nn.Module, transforms.Compose, int]:
    organ = str(organ).lower().strip()
    img_size = 384

    if organ == "flower":
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1
        model   = models.convnext_small(weights=weights)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_feat, num_classes))

    elif organ == "leaf":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model   = models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_feat, num_classes))

    elif organ == "fruit":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model   = models.efficientnet_b0(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(0.30), nn.Linear(in_feat, num_classes))

    elif organ == "bark":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model   = models.convnext_tiny(weights=weights)
        in_feat = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(0.25), nn.Linear(in_feat, num_classes))
        
    else:
        raise ValueError(f"未知器官: {organ}")

    val_tf = build_eval_tf(organ, img_size)
    return model.to(DEVICE).eval(), val_tf, img_size

def _infer_last_linear(module: nn.Module) -> nn.Linear | None:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last

def assert_state_dict_compat(model: nn.Module, sd: Dict[str, torch.Tensor], num_classes: int):
    try:
        keys = [k for k in sd.keys() if k.endswith("weight") and sd[k].dim() == 2]
        if keys:
            w = sd[keys[-1]]
            if w.size(0) != num_classes:
                raise AssertionError(f"state_dict 最后一层 out_features={w.size(0)} 与 num_classes={num_classes} 不一致")
    except Exception:
        head = _infer_last_linear(model)
        if head is not None and getattr(head, 'out_features', None) != num_classes:
            raise AssertionError(
                f"模型头部 out_features={head.out_features} 与 num_classes={num_classes} 不一致")

def load_clip_model():
    """加载CLIP模型用于植物检测和病虫害检测"""
    global clip_model, preprocess
    
    if clip_model is None:
        print("🔄 加载CLIP模型...")
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        clip_model = clip_model.to(DEVICE).eval()
        print("✅ CLIP模型加载完成")

def load_organ_classifier():
    """加载器官分类器（使用OrgansHier架构）"""
    global organ_model, organ_transform, id2organ
    
    if organ_model is not None:
        return organ_model, id2organ, organ_transform, 1.0
    
    if organ_model is None:
        print("🔄 加载器官分类器...")
        
        # 加载器官分类器模型
        organ_path = MODEL_DIR / "organ_classification_mybest.pth"
        if not organ_path.exists():
            raise FileNotFoundError(f"器官分类器模型不存在: {organ_path}")
        
        # 加载模型权重
        pack = torch.load(organ_path, map_location=DEVICE)
        
        # 构建OrgansHier模型架构
        model = OrgansHier(
            backbone=pack.get("backbone", "convnext_base"),
            pretrained=False,
            drop_path_rate=pack.get("drop_path_rate", 0.3)
        ).to(DEVICE)
        
        # 加载权重
        model.load_state_dict(pack["state_dict"], strict=False)
        model = model.eval()
        organ_model = model
        
        # 加载器官标签映射
        organ2id = pack["organ2id"]
        id2organ = {v: k for k, v in organ2id.items()}
        
        # 设置变换
        cfg = pack.get("cfg", {})
        img_size = int(cfg.get("img_size", 384))
        organ_transform = transforms.Compose([
            transforms.Resize(int(img_size * 1.05)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        ])
        
        print("✅ 器官分类器加载完成")
        return model, id2organ, organ_transform, pack.get("temperature", 1.0)

def load_species_model_for_organ_cached(organ: str):
    """加载指定器官的物种分类器（使用缓存）"""
    global species_models, species_transforms, species_id2label
    
    if organ not in species_models:
        print(f"🔄 加载{organ}物种分类器...")
        
        # 模型文件路径
        model_path = MODEL_DIR / f"{organ}_species_model.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"物种分类器模型不存在: {model_path}")
        
        # 加载模型权重
        pack = torch.load(model_path, map_location=DEVICE)
        
        # 从JSON文件获取类别信息
        map_path = MODEL_DIR / f"species_local_map_{organ}.json"
        if not map_path.exists():
            raise FileNotFoundError(f"映射文件不存在: {map_path}")
        
        with open(map_path, 'r') as f:
            species_local_map = json.load(f)
        
        num_classes = len(species_local_map)
        
        # 构建模型架构
        model, val_tfm, _ = get_infer_backbone_for_organ(organ, num_classes)
        
        # 加载权重
        sd = pack if isinstance(pack, dict) and "state_dict" not in pack else pack
        assert_state_dict_compat(model, sd, num_classes)
        model.load_state_dict(sd, strict=True)
        model = model.to(DEVICE).eval()
        species_models[organ] = model
        
        # 加载标签映射：创建从local_id到latin_name的反向映射
        species_id2label[organ] = {v: k for k, v in species_local_map.items()}
        
        # 设置变换
        species_transforms[organ] = val_tfm
        
        print(f"✅ {organ}物种分类器加载完成")
        return model, val_tfm, species_id2label[organ]

def is_plant_clip(image_path: str) -> bool:
    """使用CLIP判断是否为植物"""
    load_clip_model()
    
    if not os.path.exists(image_path):
        return False
    if os.path.isdir(image_path):
        return False
    
    labels = [
        "a photo of a plant", 
        "a photo of an animal", 
        "a photo of a person", 
        "a photo of an object",
        "a photo of a bottle",
        "a photo of an object",
        "a logo or text on white background",
        "a painting or cartoon of a plant",
        "a green text on white background"
    ]
    
    with torch.no_grad():
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        text_tokens = open_clip.tokenize(labels).to(DEVICE)
        img_feat = clip_model.encode_image(image)
        txt_feat = clip_model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        logits = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1).squeeze()
    
    top_idx = logits.argmax().item()
    top_label = labels[top_idx]
    return top_label == "a photo of a plant"

def is_diseased_clip(image_paths: List[str], vote_threshold: float = 0.7) -> bool:
    """使用CLIP判断是否有病虫害"""
    load_clip_model()
    
    healthy_prompts = [
        "The leaves of healthy plants are usually bright green",
        "The leaves of healthy plants are usually full and shiny, with no obvious signs of disease or insect damage on the leaf surface",
        "Healthy plants usually have strong, straight stems that are able to support the weight of the plant",
        "Healthy plants will show vigorous growth, including sprouting new leaves, extending branches and blooming flowers",
        "Healthy plant leaves have clear veins and are not excessively curled or wrinkled",
        "A healthy plant has bright flowers with intact petals and no wilting, falling off, or diseased spots",
        "The fruit of a healthy plant is full and has no cracks, rot or lesions. The fruit skin is normal color"
    ]
    diseased_prompts = [
        "Unhealthy plant leaves or flowers will have spots or patches of different shapes, sizes and colors, such as round, oval, polygonal, wheel-shaped",
        "Unhealthy plants have curled, shrunken, twisted leaves and flowers, and misshapen and stunted flowers",
        "Tumor-like protrusions appear on the stem, such as rose cancer, and swelling occurs",
        'Soft rot, wet rot or dry rot on the stem',
        "Unhealthy plants may have holes, nicks, or signs of being eaten on their leaves and petals",
        "Unhealthy plants may have visible insects, such as aphids and spider mites. Some pests will leave spider web-like silk",
        "Leaves lose their normal green color, show yellowing symptoms, partially or completely die, and appear brown or black",
        "The petals may appear water-soaked, rotten, softened, or even completely rotten."
    ]
    
    feats = []
    for p in image_paths:
        try:
            img = preprocess(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                f = clip_model.encode_image(img)
                feats.append(f / f.norm(dim=-1, keepdim=True))
        except:
            continue
    
    if not feats:
        return False
    
    img_feat = torch.mean(torch.stack(feats), dim=0, keepdim=True)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    votes, total = 0, 0
    
    with torch.no_grad():
        for hp, dp in zip(healthy_prompts, diseased_prompts):
            toks = open_clip.tokenize([hp, dp]).to(DEVICE)
            txt_feat = clip_model.encode_text(toks)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            logit = (100 * img_feat @ txt_feat.T).softmax(dim=-1).squeeze()
            total += 1
            votes += (logit.argmax().item() == 1)
    
    return (votes / total) >= vote_threshold

@torch.no_grad()
def organ_predict_topk(
    image_paths,
    model: nn.Module,
    tfm: transforms.Compose,
    id2organ: Dict[int, str],
    temperature: float = 1.0,
    k: int = 2,
):
    """器官Top-K预测"""
    if isinstance(image_paths, (str, Path)):
        image_paths = [str(image_paths)]
    image_paths = [str(p) for p in image_paths]

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")

    xs = []
    for p in image_paths:
        with Image.open(p) as im:
            img = im.convert("RGB")
        xs.append(tfm(img))
    batch = torch.stack(xs, dim=0).to(DEVICE)

    if next(model.parameters()).device != batch.device:
        model = model.to(batch.device)

    probs = model(batch, temperature=temperature)  # [B,4]
    topk = min(k, probs.size(1))
    confs_k, ids_k = torch.topk(probs, k=topk, dim=1)
    ids_k   = ids_k.cpu().tolist()
    confs_k = confs_k.cpu().tolist()

    top1_org  = [id2organ[row[0]] for row in ids_k]
    top1_conf = [row[0] for row in confs_k]
    return ids_k, confs_k, top1_org, top1_conf

@torch.inference_mode()
def species_predict_batch_dict(image_paths: List[str], organ: str) -> List[Dict[str, float]]:
    """物种分类预测"""
    load_species_model_for_organ_cached(organ)
    
    model = species_models[organ]
    transform = species_transforms[organ]
    id2label = species_id2label[organ]
    
    if isinstance(image_paths, (str, Path)):
        image_paths = [str(image_paths)]
    xs = []
    for p in image_paths:
        with Image.open(p) as im:
            img = im.convert("RGB")
        xs.append(transform(img))
    batch = torch.stack(xs, dim=0).to(DEVICE)
    logits = model(batch)
    probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, K_local]

    out: List[Dict[str, float]] = []
    for row in probs:
        d = { id2label[i]: float(row[i]) for i in range(len(row)) }
        out.append(d)
    return out

def predict_species_for_batch_images_soft(
    image_paths,
    organ_topk: int = 2,
    tau_min: float = 0.10,
    alpha_sharpen: float = 1.5,
) -> Dict:
    """软路由物种预测（来自organ_test.ipynb）"""
    if isinstance(image_paths, (str, Path)):
        image_paths = [str(image_paths)]
    image_paths = [str(p) for p in image_paths]

    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")

    # 1) 器官 Top-k
    organ_model, id2organ, organ_tfm, organ_T = load_organ_classifier()
    ids_k, confs_k, top1_org_text, top1_conf = organ_predict_topk(
        image_paths, organ_model, organ_tfm, id2organ, temperature=organ_T, k=organ_topk
    )

    # 2) 软路由权重（下限 + 幂次 + 归一化）
    per_image_route = []
    for i in range(len(image_paths)):
        organ_ids = ids_k[i]   # [k]
        organ_ps  = confs_k[i] # [k]
        w = np.array([max(float(p), tau_min) for p in organ_ps], dtype=np.float32)
        w = w ** float(alpha_sharpen)
        w = w / (w.sum() + 1e-12)
        sel = list(zip(organ_ids, w.tolist()))  # [(organ_id, weight), ...]
        per_image_route.append(sel)

    # 3) 分桶：按器官聚合需要推断的图片索引
    buckets: Dict[str, List[int]] = {}
    for i, sels in enumerate(per_image_route):
        for oid, _ in sels:
            organ_txt = id2organ[int(oid)]
            buckets.setdefault(organ_txt, []).append(i)

    # 4) 每个器官桶一次性跑种类头
    sp_probs: Dict[str, List[Dict[str,float]]] = {}
    for organ_txt, idxs in buckets.items():
        paths = [image_paths[j] for j in idxs]
        sp_probs[organ_txt] = species_predict_batch_dict(paths, organ_txt)

    # 5) 融合：score(s) = sum_o w_o * p(s|o)
    final = []
    for i, imgp in enumerate(image_paths):
        agg: Dict[str, float] = {}
        for oid, w in per_image_route[i]:
            organ_txt = id2organ[int(oid)]
            idx_in_bucket = buckets[organ_txt].index(i)
            d = sp_probs[organ_txt][idx_in_bucket]
            for s, p in d.items():
                agg[s] = agg.get(s, 0.0) + w * p

        if not agg:
            final.append({
                "image": imgp,
                "organ_top1": top1_org_text[i],
                "organ_top1_conf": round(float(top1_conf[i]), 4),
                "final_species": None,
                "final_conf": 0.0,
                "top3": []
            })
            continue

        # 归一化 & Top3
        z = sum(agg.values()) + 1e-12
        for k in list(agg.keys()):
            agg[k] = agg[k] / z
        top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:3]

        final.append({
            "image": imgp,
            "organ_top1": top1_org_text[i],
            "organ_top1_conf": round(float(top1_conf[i]), 4),
            "final_species": top[0][0],
            "final_conf": round(float(top[0][1]), 4),
            "top3": [(k, round(float(v), 4)) for k, v in top]
        })

    # 6) 可选：全批"投票"
    vote: Dict[str, float] = {}
    for item in final:
        if item["final_species"] is not None:
            vote[item["final_species"]] = vote.get(item["final_species"], 0.0) + item["final_conf"]
    tot = sum(vote.values()) + 1e-12
    if tot > 0:
        vote = {k: round(v/tot, 4) for k, v in vote.items()}
        final_best = max(vote.items(), key=lambda x: x[1])[0]
    else:
        vote = {}
        final_best = None

    return {
        "final_species": final_best,
        "vote_scores": vote,
        "details": final
    }

def download_image(url: str, temp_dir: str = "uploads") -> str:
    """下载网络图片并保存到临时文件"""
    try:
        print(f"🌐 正在下载: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(
            url, 
            timeout=30, 
            headers=headers, 
            allow_redirects=True,
            verify=True,
            stream=True
        )
        
        if response.status_code != 200:
            print(f"❌ HTTP错误 {response.status_code}: {url}")
            return None
        
        # 生成唯一文件名
        file_extension = os.path.splitext(url.split('?')[0])[1] or '.jpg'
        filename = f"{uuid.uuid4()}{file_extension}"
        filepath = os.path.join(temp_dir, filename)
        
        # 确保目录存在
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存文件
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ 下载成功: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def predict_plant_full(image_paths: List[str]) -> Dict:
    """完整的植物识别流程，输出逻辑与 plant_api/inference.py 对齐（不改动路由与分类器）。"""
    try:
        # 输入标准化与批量上限（与 v1 对齐）
        if isinstance(image_paths, (str, Path)):
            image_paths = [str(image_paths)]
        if not image_paths:
            return {"code": 400, "msg": "错误：输入图像列表为空", "data": {}}

        max_batch = 3
        image_paths = image_paths[:max_batch]

        # 拒识策略阈值（与 v1 保持一致的默认值）
        prob_threshold = 0.42
        ent_threshold = 1.5

        # 1) 植物性检测（多数规则）
        is_plant_flags = [is_plant_clip(p) for p in image_paths]
        num_plant = sum(is_plant_flags)
        if num_plant <= (len(image_paths) // 2):
            return {
                "code": 200, "msg": "成功",
                "data": {"is_plant": False, "is_database": False, "plants": [], "is_infected": False}
            }

        # 仅对判为植物的图片进行物种识别
        valid_paths = [p for p, f in zip(image_paths, is_plant_flags) if f]

        # 2) 物种识别（软路由不改动）
        infer_out = predict_species_for_batch_images_soft(
            valid_paths,
            organ_topk=2,
            tau_min=0.10,
            alpha_sharpen=1.5
        )

        # 3) 基于阈值的拒识与多图统计（对齐 v1 行为）
        known_results = []  # 可用预测（通过拒识）
        all_top1_species = []
        species_to_scores = {}
        species_to_images = {}

        details = infer_out.get("details", [])
        for path, detail in zip(valid_paths, details):
            final_species = detail.get("final_species")
            final_conf = float(detail.get("final_conf", 0.0))
            top3_pairs = detail.get("top3", [])  # [(latin_name, prob), ...]

            if final_species is None:
                all_top1_species.append("unknown")
                continue

            # 近似熵：使用 top3 + 剩余质量作为一类
            sum_top3 = float(sum(v for _, v in top3_pairs))
            probs_for_entropy = [float(v) for _, v in top3_pairs]
            rest_mass = max(1.0 - sum_top3, 0.0)
            if rest_mass > 0:
                probs_for_entropy.append(rest_mass)
            p = np.array(probs_for_entropy, dtype=np.float32)
            p = p / (float(p.sum()) + 1e-12)
            ent = float(-(p * np.log(p + 1e-12)).sum())

            # 拒识：低置信或高熵 → 认为是库外
            if (final_conf < prob_threshold) or (ent > ent_threshold):
                all_top1_species.append("unknown")
                continue

            candidates = [{"latin_name": s, "confidence": round(float(v), 4)} for s, v in top3_pairs]

            all_top1_species.append(final_species)
            known_results.append({
                "path": path,
                "species": final_species,
                "confidence": round(final_conf, 4),
                "candidates": candidates,
            })
            species_to_scores.setdefault(final_species, []).append(final_conf)
            species_to_images.setdefault(final_species, []).append(path)

        # 若无任何可识别结果
        if not known_results:
            return {
                "code": 200, "msg": "成功",
                "data": {"is_plant": True, "is_database": False, "plants": [], "is_infected": False}
            }

        # 单图：直接返回 TopK 候选 + 病虫害
        if len(image_paths) == 1:
            r = known_results[0]
            is_infected = is_diseased_clip([r["path"]], vote_threshold=0.5)
            return {
                "code": 200, "msg": "成功",
                "data": {
                    "is_plant": True,
                    "is_database": True,
                    "plants": r["candidates"],
                    "is_infected": bool(is_infected)
                }
            }

        # 多图：投票 + 平均置信度
        counter = collections.Counter([s for s in all_top1_species if s != "unknown"])
        if not counter:
            return {
                "code": 200, "msg": "成功",
                "data": {"is_plant": True, "is_database": False, "plants": [], "is_infected": False}
            }

        top3 = counter.most_common(3)
        plants_list = []
        for name, votes in top3:
            scores = species_to_scores.get(name, [])
            if scores:
                conf = round(float(sum(scores) / len(scores)), 4)
            else:
                conf = round(votes / len(all_top1_species) * 0.8, 4)
            plants_list.append({"latin_name": name, "confidence": conf})

        top1_species = top3[0][0]
        disease_paths = species_to_images.get(top1_species, [])
        is_infected = is_diseased_clip(disease_paths, vote_threshold=0.5)

        return {
            "code": 200, "msg": "成功",
            "data": {
                "is_plant": True,
                "is_database": True,
                "plants": plants_list,
                "is_infected": bool(is_infected)
            }
        }

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            "code": 500,
            "msg": f"处理失败: {str(e)}",
            "data": {}
        }

# 兼容性函数
def predict_plant(image_path: str) -> Dict:
    """单张图片预测"""
    return predict_plant_full([image_path])

def batch_predict_species_with_disease(image_paths: List[str]) -> Dict:
    """批量预测（与plant_api接口兼容）"""
    return predict_plant_full(image_paths)
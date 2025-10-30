import os
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


def index_image_paths_fast(root_dir, suffixes=(".jpg", ".jpeg", ".png")):
    records = []
    for label in tqdm(os.scandir(root_dir), desc="索引植物类别"):
        if not label.is_dir():
            continue
        label_name = label.name
        try:
            for file in os.scandir(label.path):
                if file.is_file() and file.name.lower().endswith(suffixes):
                    organ = file.name.split("_")[0].lower()
                    records.append((label_name, organ, file.path))
        except Exception as e:
            print(f"跳过 {label_name}: {e}")
            continue
    return pd.DataFrame(records, columns=["label", "organ", "image_path"])


def main():
    parser = argparse.ArgumentParser(description="扫描植物图片目录并生成标准 CSV")
    parser.add_argument("--plant-dir", required=True, help="植物图片根目录（按 label 子目录组织）")
    parser.add_argument("--out-csv", required=True, help="输出 CSV 路径")
    parser.add_argument("--country", required=True, help="国家缩写（如 CN/BR），用于命令链路统一")
    args = parser.parse_args()

    df = index_image_paths_fast(args["plant-dir"]) if isinstance(args, dict) else index_image_paths_fast(args.plant_dir)

    organ_encoder = LabelEncoder()
    species_encoder = LabelEncoder()
    df["organ_id"] = organ_encoder.fit_transform(df["organ"])
    df["species"] = species_encoder.fit_transform(df["label"])

    print(f"共索引到 {len(df)} 张图片")
    print("器官种类：", sorted(df["organ"].unique()))

    out_csv = args["out-csv"] if isinstance(args, dict) else args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    # 输出标准列顺序
    cols = ["image_path", "organ", "organ_id", "label", "species"]
    df = df[cols]
    df.to_csv(out_csv, index=False)
    print(f"✅ 已保存 CSV: {out_csv}")


if __name__ == "__main__":
    main()


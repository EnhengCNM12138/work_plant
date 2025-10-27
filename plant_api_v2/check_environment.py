"""
环境检查脚本 - 检查模型文件和依赖
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查Python版本...")
    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("  ✅ Python版本符合要求")
        return True
    else:
        print("  ❌ Python版本过低，需要3.10+")
        return False

def check_dependencies():
    """检查依赖包"""
    print("=" * 60)
    print("检查依赖包...")
    
    required_packages = [
        'torch',
        'torchvision',
        'fastapi',
        'uvicorn',
        'open_clip',
        'timm',
        'PIL',
        'numpy',
        'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'open_clip':
                __import__('open_clip')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (未安装)")
            missing.append(package)
    
    if missing:
        print(f"\n  缺少依赖: {', '.join(missing)}")
        print(f"  请运行: pip install -r requirements.txt")
        return False
    else:
        print("  ✅ 所有依赖已安装")
        return True

def check_cuda():
    """检查CUDA"""
    print("=" * 60)
    print("检查CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️  CUDA不可用，将使用CPU模式")
            return True
    except Exception as e:
        print(f"  ❌ CUDA检查失败: {e}")
        return False

def check_model_files():
    """检查模型文件"""
    print("=" * 60)
    print("检查模型文件...")
    
    model_dir = Path("../real_data")
    
    if not model_dir.exists():
        print(f"  ❌ 模型目录不存在: {model_dir}")
        return False
    
    required_files = [
        "organ_classification_mybest.pth",
        "bark_species_model.pth",
        "flower_species_model.pth",
        "fruit_species_model.pth",
        "leaf_species_model.pth",
        "organ_classes.json",
        "species_local_map_bark.json",
        "species_local_map_flower.json",
        "species_local_map_fruit.json",
        "species_local_map_leaf.json",
        "species_local2global_bark.json",
        "species_local2global_flower.json",
        "species_local2global_fruit.json",
        "species_local2global_leaf.json",
    ]
    
    missing = []
    for file in required_files:
        filepath = model_dir / file
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {file} ({size:.2f} MB)")
        else:
            print(f"  ❌ {file} (缺失)")
            missing.append(file)
    
    if missing:
        print(f"\n  缺少模型文件: {len(missing)}个")
        return False
    else:
        print("  ✅ 所有模型文件完整")
        return True

def check_api_files():
    """检查API文件"""
    print("=" * 60)
    print("检查API文件...")
    
    required_files = [
        "app.py",
        "inference.py",
        "requirements.txt",
        "start_api.sh"
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (缺失)")
            missing.append(file)
    
    if missing:
        print(f"\n  缺少API文件: {', '.join(missing)}")
        return False
    else:
        print("  ✅ 所有API文件完整")
        return True

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("植物识别API v2 - 环境检查")
    print("=" * 60 + "\n")
    
    results = []
    
    # 检查Python版本
    results.append(("Python版本", check_python_version()))
    
    # 检查依赖
    results.append(("依赖包", check_dependencies()))
    
    # 检查CUDA
    results.append(("CUDA", check_cuda()))
    
    # 检查模型文件
    results.append(("模型文件", check_model_files()))
    
    # 检查API文件
    results.append(("API文件", check_api_files()))
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 环境检查通过！可以启动API服务。")
        print("\n启动命令:")
        print("  bash start_api.sh")
        print("  或")
        print("  python -m uvicorn app:app --host 0.0.0.0 --port 8000")
        return 0
    else:
        print("\n❌ 环境检查失败，请解决上述问题后重试。")
        return 1

if __name__ == "__main__":
    sys.exit(main())


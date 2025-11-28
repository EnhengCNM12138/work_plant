"""
植物识别API接口 - FastAPI实现
基于organ_test.ipynb的完整推理流程
"""

from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel, Field
import os

#from inference import predict_plant_full, download_image
from inference import predict_plant_full, download_image, get_devices_string

import time

import os
import logging
from logging.handlers import RotatingFileHandler



# ===== 日志配置：写到 /var/log/plant_api.log，同时打印到控制台 =====
logger = logging.getLogger("plant_api")
logger.setLevel(logging.INFO)

if not logger.handlers:  # 避免 uvicorn reload 时重复添加 handler
    log_path = "./plant_api.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB 自动轮转
        backupCount=10,
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台也打日志（方便你现在看）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)



app = FastAPI(
    title="植物识别API v2",
    description="支持远程图片URL的植物识别服务（器官+种类分类器）",
    version="2.0.0"
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Pydantic模型定义 - JSON格式
class URLRequest(BaseModel):
    image_urls: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=10,
        description="图片URL列表，最多10个",
        example=[
            "https://example.com/plant1.jpg",
            "https://example.com/plant2.jpg"
        ]
    )


@app.post("/predict/")
async def predict_from_urls_json(request: URLRequest):
    """
    通过远程图片URL进行植物识别 (JSON格式)
    
    支持：
    - 单个或多个URL（最多10个）
    - JSON格式输入
    - 自动下载图片并进行识别
    - 返回植物种类、置信度等信息
    """
    image_urls = request.image_urls
    
    downloaded_paths = []
    failed_urls = []
    
    try:
        t0 = time.time() # 开始时间
        print(f"🔍 接收到 {len(image_urls)} 个URL: {image_urls}")
        
        # 下载所有图片
        t_dl_start = time.time()               # ★ 下载起点
        for url in image_urls:
            print(f"📥 下载图片: {url}")
            filepath = download_image(url, UPLOAD_DIR)
            if filepath:
                downloaded_paths.append(filepath)
                print(f"✅ 下载成功: {filepath}")
            else:
                failed_urls.append(url)
                print(f"❌ 下载失败: {url}")
        t_dl_end = time.time()               # ★ 下载终点
        print(f"🌐 下载完成: {t_dl_end - t_dl_start:.2f}秒")

        if not downloaded_paths:
            return JSONResponse(
                status_code=400,
                content={
                    "code": 400,
                    "msg": "所有图片下载失败",
                    "data": {"failed_urls": failed_urls}
                }
            )
        
        print(f"🔬 开始推理 {len(downloaded_paths)} 张图片...")

        t_inf_start = time.time()              # ★ 推理起点
        result = predict_plant_full(downloaded_paths)
        t_inf_end = time.time()                # ★ 推理结束

        t_end = time.time()
        devices = get_devices_string()
        logger.info(
            "⌚️ [TIMER] /predict(json): "
            f"download={t_dl_end - t_dl_start:.3f}s, "
            f"infer={t_inf_end - t_inf_start:.3f}s, "
            f"other={t_end - t_inf_end + t0 - t0:.3f}s, "
            f"total={t_end - t0:.3f}s, "
            f"💾 [DEVICES]:({devices})"
        )

        # 添加下载失败信息
        if failed_urls:
            result["failed_urls"] = failed_urls
        
        print(f"✅ 推理完成: {result['code']}")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "msg": f"服务器错误: {str(e)}",
                "data": {}
            }
        )
    finally:
        # 清理下载的文件
        for path in downloaded_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🗑️ 清理文件: {path}")
            except:
                pass


@app.post("/predict/form")
async def predict_from_urls_form(image_urls: List[str] = Form(..., description="图片URL列表，每行一个URL，最多10个")):
    """
    通过远程图片URL进行植物识别 (Form格式，SwaggerUI友好)
    
    支持：
    - 单个或多个URL（最多10个）  
    - Form表单格式输入
    - 在SwaggerUI中显示为表单字段
    - 自动下载图片并进行识别
    - 返回植物种类、置信度等信息
    """
    
    # 处理可能的多种输入格式
    processed_urls = []
    for url_item in image_urls:
        # 处理可能包含多个URL的字符串（逗号分隔）
        if ',' in url_item:
            urls = [u.strip() for u in url_item.split(',') if u.strip()]
            processed_urls.extend(urls)
        # 处理换行符分隔的URL
        elif '\n' in url_item:
            urls = [u.strip() for u in url_item.split('\n') if u.strip()]
            processed_urls.extend(urls)
        else:
            processed_urls.append(url_item.strip())
    
    # 去重并限制数量
    processed_urls = list(dict.fromkeys(processed_urls))  # 去重保持顺序
    if len(processed_urls) > 10:
        return JSONResponse(
            content={
                "code": 400,
                "msg": "错误：图片URL数量不能超过10个",
                "data": {}
            }
        )
    
    if not processed_urls:
        return JSONResponse(
            content={
                "code": 400,
                "msg": "错误：未提供任何有效的图片URL",
                "data": {}
            }
        )
    
    downloaded_paths = []
    failed_urls = []
    
    try:
        t0 = time.time() # 开始时间
        print(f"🔍 接收到 {len(processed_urls)} 个URL: {processed_urls}")
        
        # 下载图片
        t_dl_start = time.time()
        for url in processed_urls:
            print(f"📥 下载图片: {url}")
            result = download_image(url, UPLOAD_DIR)
            if result:
                downloaded_paths.append(result)
                print(f"✅ 下载成功: {result}")
            else:
                failed_urls.append(url)
                print(f"❌ 下载失败: {url}")
        t_dl_end = time.time()
        
        if not downloaded_paths:
            return JSONResponse(
                content={
                    "code": 400,
                    "msg": f"错误：所有图片下载失败。失败的URL: {failed_urls}",
                    "data": {}
                }
            )
        
        print(f"🔬 开始推理 {len(downloaded_paths)} 张图片...")

        t_inf_start = time.time()
        result = predict_plant_full(downloaded_paths)
        t_inf_end = time.time()

        t_end = time.time()
        devices = get_devices_string()
        logger.info(
            "⌚️ [TIMER] /predict(json): "
            f"download={t_dl_end - t_dl_start:.3f}s, "
            f"infer={t_inf_end - t_inf_start:.3f}s, "
            f"other={t_end - t_inf_end + t0 - t0:.3f}s, "
            f"total={t_end - t0:.3f}s, "
            f"💾 [DEVICES]:({devices})"
        )

        
        # 在结果中添加下载信息
        if failed_urls:
            if "msg" in result:
                result["msg"] += f"（注意：{len(failed_urls)}个URL下载失败）"
        
        print(f"✅ 推理完成: {result['code']}")
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ 服务器错误: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "code": 500,
                "msg": f"服务器错误：{str(e)}",
                "data": {}
            }
        )
    finally:
        # 清理临时文件
        for file_path in downloaded_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️ 清理文件: {file_path}")
            except:
                pass


@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "植物识别API v2",
        "version": "2.0.0",
        "endpoints": {
            "predict_json": "/predict/",
            "predict_form": "/predict/form",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


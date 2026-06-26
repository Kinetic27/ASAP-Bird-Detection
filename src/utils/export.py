
# src/export.py

import os
from ultralytics import YOLO
from src.utils.config import PATCH_SIZE, TRT_BATCH_SIZE, TRT_DYNAMIC_SHAPES

def export_tensorrt(model_path, device="0", imgsz=PATCH_SIZE, batch=TRT_BATCH_SIZE, dynamic=TRT_DYNAMIC_SHAPES):
    """
    Export a YOLO model to TensorRT engine format with optimized settings.
    
    Args:
        model_path (str): Path to the .pt model file.
        device (str): Device to use for export (e.g., "0").
    """
    print(f"Starting TensorRT export for {model_path}...")
    print(f"Settings: imgsz={imgsz}, batch={batch}, format=engine, half=True, dynamic={dynamic}")
    
    model = YOLO(model_path)
    
    # Export arguments matching the paper-runtime TensorRT engines.
    # Use batch=8 for the 1280px / 8-patch paper path. Static engines
    # are fastest and lowest-memory for that fixed shape; dynamic engines
    # remain useful when varying runtime batch/shape is required.
    metrics = model.export(
        format="engine",
        device=device,
        half=True,
        imgsz=imgsz,
        dynamic=dynamic,
        batch=batch,
        exist_ok=True # Prevent increments in runs/detect if logs are enabled
    )
    
    print(f"Export completed. Metrics: {metrics}")
    
    # Expected output path
    base, ext = os.path.splitext(model_path)
    engine_path = base + ".engine"
    
    if os.path.exists(engine_path):
        print(f"Engine saved to: {engine_path}")
        return engine_path
    else:
        print("Warning: Engine file not found at expected location.")
        return None

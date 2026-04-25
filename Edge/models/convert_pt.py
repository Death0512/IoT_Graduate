from ultralytics import YOLO
import os

def main():
    models_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Download yolo11n.pt and export to ONNX (Static Batch 1 is fine for Primary)
    print("Downloading yolo11n.pt...")
    model = YOLO("yolo11n.pt")
    # For Primary detector, static batch 1 is usually okay and slightly faster/simpler.
    # But if we want to be safe, dynamic is better. Let's stick to valid static for now.
    print("Exporting yolo11n.pt to ONNX (Static)...")
    model.export(format="onnx", opset=12, dynamic=False) 
    
    if os.path.exists("yolo11n.pt") and not os.path.exists(os.path.join(models_dir, "yolo11n.pt")):
        os.rename("yolo11n.pt", os.path.join(models_dir, "yolo11n.pt"))
    if os.path.exists("yolo11n.onnx") and not os.path.exists(os.path.join(models_dir, "yolo11n.onnx")):
        os.rename("yolo11n.onnx", os.path.join(models_dir, "yolo11n.onnx"))

    # 2. Export lpd.pt to ONNX (DYNAMIC for Batching)
    lpd_path = os.path.join(models_dir, "lpd.pt")
    if os.path.exists(lpd_path):
        print(f"Exporting {lpd_path} to ONNX (Dynamic)...")
        try:
            model_lpd = YOLO(lpd_path)
            # Enable dynamic shapes for variable batch size
            model_lpd.export(format="onnx", opset=12, dynamic=True)
            
            if os.path.exists("lpd.onnx"):
                # Overwrite existing onnx in models/
                target = os.path.join(models_dir, "lpd.onnx")
                if os.path.exists(target):
                    os.remove(target)
                os.rename("lpd.onnx", target)
                
        except Exception as e:
            print(f"Failed to export lpd.pt: {e}")
    else:
        print("lpd.pt not found")
        
    # 3. Export lpr.pt (assuming it exists, otherwise skipped) - usually handled separately or already exists.
    # The user only asked for yolo11n and lpd in the original request, but let's check lpr too.
    # The previous turn showed lpr.onnx exists (57MB) but lpr.pt wasn't listed or I missed it.
    # Actually lpr.onnx was there. If it's static, we might need to re-export if possible.
    # But I don't see lpr.pt in the file list. So I can only work with lpd and yolo.
    # If lpr.onnx is static batch 1, and config wants 16, it might be an issue.
    # But usually LPR engines are provided or custom trained.
    # Let's assume lpr.onnx is what it is.

if __name__ == "__main__":
    main()

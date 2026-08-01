import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import io, base64
from PIL import Image as PILImage

class YOLOONNXDetector:
    """YOLO aircraft detector using ONNX Runtime (no PyTorch needed)."""
    
    # COCO class 4 = airplane
    AIRCRAFT_CLASS_ID = 4
    CLASS_NAMES = [
        'person','bicycle','car','motorcycle','airplane','bus','train',
        'truck','boat','traffic light','fire hydrant','stop sign',
        'parking meter','bench','bird','cat','dog','horse','sheep',
        'cow','elephant','bear','zebra','giraffe','backpack','umbrella',
        'handbag','tie','suitcase','frisbee','skis','snowboard',
        'sports ball','kite','baseball bat','baseball glove','skateboard',
        'surfboard','tennis racket','bottle','wine glass','cup','fork',
        'knife','spoon','bowl','banana','apple','sandwich','orange',
        'broccoli','carrot','hot dog','pizza','donut','cake','chair',
        'couch','potted plant','bed','dining table','toilet','tv',
        'laptop','mouse','remote','keyboard','cell phone','microwave',
        'oven','toaster','sink','refrigerator','book','clock','vase',
        'scissors','teddy bear','hair drier','toothbrush'
    ]
    
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.input_size = 640
        
        if model_path and Path(model_path).exists():
            self.session = ort.InferenceSession(
                model_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        else:
            # Download yolov8n.onnx if not present
            self.session = self._load_or_download()
        
        self.input_name = self.session.get_inputs()[0].name
        self.model_name = "yolov8n-onnx"
    
    def _load_or_download(self):
        import urllib.request
        onnx_path = Path("ml_core/artifacts/yolov8n.onnx")
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if not onnx_path.exists():
            print("Downloading yolov8n.onnx...")
            # Export from ultralytics if available, else use pretrained
            try:
                from ultralytics import YOLO
                model = YOLO("yolov8n.pt")
                model.export(format="onnx", imgsz=640)
                import shutil
                shutil.move("yolov8n.onnx", str(onnx_path))
            except ImportError:
                # Download ONNX directly
                url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
                urllib.request.urlretrieve(url, str(onnx_path))
        return ort.InferenceSession(
            str(onnx_path),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    
    def preprocess(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (nw, nh))
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        return blob, scale, h, w
    
    def postprocess(self, outputs, scale, orig_h, orig_w):
        predictions = outputs[0][0].T  # [8400, 84]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        
        # Filter to aircraft class only
        mask = (class_ids == self.AIRCRAFT_CLASS_ID) & (confidences >= self.conf_threshold)
        boxes = boxes[mask]
        confidences = confidences[mask]
        
        detections = []
        for box, conf in zip(boxes, confidences):
            cx, cy, bw, bh = box
            x1 = int((cx - bw/2) / scale)
            y1 = int((cy - bh/2) / scale)
            x2 = int((cx + bw/2) / scale)
            y2 = int((cy + bh/2) / scale)
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, float(conf)))
        return detections
    
    def detect(self, img_bgr: np.ndarray):
        blob, scale, h, w = self.preprocess(img_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return self.postprocess(outputs, scale, h, w)


class AircraftDetectionService:
    def __init__(self):
        self.detector = YOLOONNXDetector(conf_threshold=0.25)
        self._classifier = None
    
    @property
    def classifier(self):
        if self._classifier is None:
            from app.services.vit_service import get_aircraft_classifier
            self._classifier = get_aircraft_classifier()
        return self._classifier
    
    def detect_and_classify(self, image_bgr, conf_threshold=0.25, country="USA"):
        h, w = image_bgr.shape[:2]
        raw_detections = self.detector.detect(image_bgr)
        
        detections = []
        annotated = image_bgr.copy()
        
        for (x1, y1, x2, y2, det_conf) in raw_detections:
            pad = 10
            cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
            cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
            crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
            if crop_bgr.size == 0:
                continue
            
            try:
                cls_result = self.classifier.classify(crop_bgr)
                class_name = cls_result.class_name
                class_conf = cls_result.confidence
                origin_country = cls_result.origin_country
                from app.services.geopolitics import classify_friend_foe
                friend_or_foe = classify_friend_foe(country, origin_country)
            except Exception:
                class_name = "Aircraft"
                class_conf = det_conf
                origin_country = "Unknown"
                friend_or_foe = "NEUTRAL"
            
            color = {"FRIEND":(0,255,100),"FOE":(0,50,255),"NEUTRAL":(255,200,0)}.get(friend_or_foe,(200,200,200))
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            label = f"{class_name} {class_conf*100:.0f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, (x1,y1-label_size[1]-8), (x1+label_size[0]+4,y1), color, -1)
            cv2.putText(annotated, label, (x1+2,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            PILImage.fromarray(crop_rgb).save(buf, format="PNG")
            crop_b64 = base64.b64encode(buf.getvalue()).decode()
            
            detections.append({
                "bbox": {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
                "detection_confidence": round(det_conf,4),
                "class_name": class_name,
                "class_confidence": round(class_conf,4),
                "origin_country": origin_country,
                "friend_or_foe": friend_or_foe,
                "crop_base64": crop_b64,
            })
        
        ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        buf2 = io.BytesIO()
        PILImage.fromarray(ann_rgb).save(buf2, format="JPEG", quality=92)
        ann_b64 = base64.b64encode(buf2.getvalue()).decode()
        
        return {
            "detections": detections,
            "annotated_image_base64": ann_b64,
            "total_aircraft": len(detections),
            "model_used": "yolov8n-onnx + convnext_small",
        }


_detection_service: AircraftDetectionService | None = None


def get_detection_service() -> AircraftDetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = AircraftDetectionService()
    return _detection_service

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import io, base64
from pathlib import Path

class AircraftDetectionService:
    """
    Two-stage pipeline:
    1. YOLOv8 detects aircraft bounding boxes (pretrained COCO)
    2. ConvNeXt classifies each cropped box
    """
    
    def __init__(self):
        # Load pretrained YOLOv8 - downloads automatically on first run
        self.detector = YOLO("yolov8n.pt")  # nano model, fast
        self.aircraft_class_id = 4  # COCO class 4 = airplane
        self._classifier = None
    
    @property
    def classifier(self):
        if self._classifier is None:
            from app.services.vit_service import get_aircraft_classifier
            self._classifier = get_aircraft_classifier()
        return self._classifier
    
    def detect_and_classify(
        self, 
        image_bgr: np.ndarray,
        conf_threshold: float = 0.25,
        country: str = "USA"
    ) -> dict:
        """
        Returns:
        {
          detections: [
            {
              bbox: {x1, y1, x2, y2},
              detection_confidence: float,
              class_name: str,
              class_confidence: float,
              origin_country: str,
              friend_or_foe: str,
              crop_base64: str  # base64 PNG of the cropped aircraft
            }
          ],
          annotated_image_base64: str,  # full image with boxes drawn
          total_aircraft: int,
          model_used: str
        }
        """
        h, w = image_bgr.shape[:2]
        
        # Stage 1: YOLO detection
        results = self.detector(image_bgr, classes=[self.aircraft_class_id], 
                                conf=conf_threshold, verbose=False)
        
        detections = []
        annotated = image_bgr.copy()
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                det_conf = float(box.conf[0])
                
                # Add padding around crop
                pad = 10
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(w, x2 + pad)
                cy2 = min(h, y2 + pad)
                
                crop_bgr = image_bgr[cy1:cy2, cx1:cx2]
                if crop_bgr.size == 0:
                    continue
                
                # Stage 2: ConvNeXt classification on crop
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
                
                # Color by friend/foe
                color = {
                    "FRIEND": (0, 255, 100),
                    "FOE": (0, 50, 255),
                    "NEUTRAL": (255, 200, 0)
                }.get(friend_or_foe, (200, 200, 200))
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name} {class_conf*100:.0f}%"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, 
                              (x1, y1 - label_size[1] - 8),
                              (x1 + label_size[0] + 4, y1),
                              color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Encode crop
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                buf = io.BytesIO()
                pil_crop.save(buf, format="PNG")
                crop_b64 = base64.b64encode(buf.getvalue()).decode()
                
                detections.append({
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "detection_confidence": round(det_conf, 4),
                    "class_name": class_name,
                    "class_confidence": round(class_conf, 4),
                    "origin_country": origin_country,
                    "friend_or_foe": friend_or_foe,
                    "crop_base64": crop_b64
                })
        
        # Encode full annotated image
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_ann = Image.fromarray(annotated_rgb)
        buf2 = io.BytesIO()
        pil_ann.save(buf2, format="JPEG", quality=92)
        annotated_b64 = base64.b64encode(buf2.getvalue()).decode()
        
        return {
            "detections": detections,
            "annotated_image_base64": annotated_b64,
            "total_aircraft": len(detections),
            "model_used": "yolov8n + convnext_small"
        }


_detection_service: AircraftDetectionService | None = None


def get_detection_service() -> AircraftDetectionService:
    global _detection_service
    if _detection_service is None:
        _detection_service = AircraftDetectionService()
    return _detection_service

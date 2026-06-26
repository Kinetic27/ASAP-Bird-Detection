import cv2
import numpy as np
from .config import VIS_THICKNESS, VIS_FONT_SCALE

_LEGEND_CACHE = None

def _get_legend_panel():
    global _LEGEND_CACHE
    if _LEGEND_CACHE is not None:
        return _LEGEND_CACHE
    
    panel_w, panel_h = 350, 150
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Static background for panel
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(panel, (0, 0), (panel_w, panel_h), (255, 255, 255), 2)

    # Confidence Legend
    cv2.putText(panel, "Conf:", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Legend Bar (Yellow -> Red)
    for i in range(101):
        score_val = i / 100.0
        g = int(255 * (1 - score_val))
        col = (0, g, 255)
        cv2.line(panel, (100 + i*2, 95), (100 + i*2, 120), col, 2)
    
    cv2.putText(panel, "0.0", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(panel, "1.0", (280, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    _LEGEND_CACHE = panel
    return panel


def draw_visuals(frame, predictions):
    """
    Draw bounding boxes on the frame.
    
    Args:
        frame (numpy.ndarray): The image frame.
        predictions (list): List of detections [x1, y1, x2, y2, conf, cls].
    """
    h, w = frame.shape[:2]
    bird_count = len(predictions)

    for bbox in predictions:
        try:
            # Take only the first 6 elements regardless of what's passed
            b = list(bbox)[:6]
            x1, y1, x2, y2, score, cls = b
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        except Exception as e:
            continue
        
        # Dynamic color: Yellow (Low conf) -> Red (High conf)
        # BGR: Yellow=(0, 255, 255), Red=(0, 0, 255)
        green = int(255 * (1 - score))
        color = (0, green, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, VIS_THICKNESS)
        
        # Draw score
        label = f"{score:.2f}"

        if y1 - 2 > 0:
             cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, VIS_FONT_SCALE, color, VIS_THICKNESS)
        else:
             cv2.putText(frame, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, VIS_FONT_SCALE, color, VIS_THICKNESS)

    # --- Draw Legend and Count in Top-Right ---
    panel = _get_legend_panel().copy()
    px, py = w - panel.shape[1] - 20, 20
    
    # Count (Dynamic part of panel)
    cv2.putText(panel, f"Bird Count: {bird_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Overlay panel on frame
    frame[py:py+panel.shape[0], px:px+panel.shape[1]] = panel
             
    return frame

import cv2
import json
import torch
import numpy as np
from collections import deque, Counter
import time
from preprocess import resize_frames, normalize_and_tensor, format_for_r3d
from r3d_model import get_r3d_model

MODEL_PATH            = r"C:\Users\sreej\OneDrive\Desktop\webcam2\best_model.pth"
CLASS_NAMES_PATH      = r"C:\Users\sreej\OneDrive\Desktop\webcam2\class_names.json"
DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE           = 16
PREDICT_EVERY         = 8
SMOOTH_WINDOW         = 7
CONFIDENCE_THRESHOLD  = 40.0   

with open(CLASS_NAMES_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = get_r3d_model(num_classes=len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)
print(f" Model loaded — {len(class_to_idx)} classes | Device: {DEVICE}")
print("Starting webcam... Press 'Q' or 'Esc' to quit.")

def get_confidence_color(conf):
    if conf >= 70:
        return (0, 255, 0)     
    elif conf >= 40:
        return (0, 165, 255) 
    else:
        return (0, 0, 255)     

def predict_from_buffer(frame_buffer):
    frames       = list(frame_buffer)
    resized      = resize_frames(frames)
    tensor       = normalize_and_tensor(resized)
    final_tensor = format_for_r3d(tensor).to(DEVICE)
    with torch.no_grad():
        outputs = model(final_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        top3    = probs.topk(3)
    results = []
    for prob, idx in zip(top3.values, top3.indices):
        results.append({
            "action"    : idx_to_class[idx.item()],
            "confidence": round(prob.item() * 100, 1)
        })
    return results

def run_dashboard():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not found! Try VideoCapture(1)")
        return

    frame_buf     = deque(maxlen=BUFFER_SIZE)
    smooth_buf    = deque(maxlen=SMOOTH_WINDOW)
    predictions   = [{"action": "Warming up...", "confidence": 0.0},
                     {"action": "-",             "confidence": 0.0},
                     {"action": "-",             "confidence": 0.0}]
    history       = deque(maxlen=5)
    stable_action = "Warming up..."
    fps_time      = time.time()
    fps           = 0.0
    frame_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost webcam connection!")
            break

        frame_count += 1
        if frame_count % 10 == 0:
            elapsed  = time.time() - fps_time
            fps      = 10 / elapsed if elapsed > 0 else 0
            fps_time = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buf.append(rgb)

        if len(frame_buf) == BUFFER_SIZE and frame_count % PREDICT_EVERY == 0:
            predictions = predict_from_buffer(frame_buf)

            smooth_buf.append(predictions[0]["action"])
            stable_action = Counter(smooth_buf).most_common(1)[0][0]

            if not history or history[0] != stable_action:
                if predictions[0]["confidence"] >= CONFIDENCE_THRESHOLD:
                    history.appendleft(stable_action)

        top_conf = predictions[0]["confidence"]
        if top_conf >= CONFIDENCE_THRESHOLD:
            display_action = stable_action
        else:
            display_action = "No action detected"

        cam_display = cv2.resize(frame, (480, 360))
        bar_color   = get_confidence_color(top_conf)

        cv2.rectangle(cam_display, (0, 305), (480, 360), (0, 0, 0), -1)
        cv2.circle(cam_display, (18, 335), 8, bar_color, -1)
        cv2.putText(cam_display,
                    f"{display_action}   {top_conf:.1f}%",
                    (35, 342), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, bar_color, 2)
        
        panel = np.zeros((360, 320, 3), dtype=np.uint8)

        cv2.putText(panel, "ACTION RECOGNITION",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 0), 2)
        cv2.line(panel, (10, 38), (310, 38), (80, 80, 80), 1)

        cv2.putText(panel, f"FPS: {fps:.1f}   Device: {str(DEVICE).upper()}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (180, 180, 180), 1)

        cv2.circle(panel, (14, 74), 5, (0, 255, 0), -1)
        cv2.putText(panel, ">=70%", (22, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.circle(panel, (74, 74), 5, (0, 165, 255), -1)
        cv2.putText(panel, "40-70%", (82, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        cv2.circle(panel, (140, 74), 5, (0, 0, 255), -1)
        cv2.putText(panel, "<40%", (148, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        cv2.line(panel, (10, 86), (310, 86), (80, 80, 80), 1)

        cv2.putText(panel, "TOP-3 PREDICTIONS:",
                    (10, 103), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        for i, pred in enumerate(predictions[:3]):
            y     = 115 + i * 46
            bar_w = max(int(pred["confidence"] * 2.8), 2)
            color = get_confidence_color(pred["confidence"])

            cv2.rectangle(panel, (10, y + 4), (290, y + 24), (50, 50, 50), -1)
            cv2.rectangle(panel, (10, y + 4), (10 + bar_w, y + 24), color, -1)
            short = pred["action"][:22]
            cv2.putText(panel, f"{i+1}. {short}",
                        (13, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1)
            cv2.putText(panel, f"{pred['confidence']:.1f}%",
                        (248, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1)

        cv2.line(panel, (10, 260), (310, 260), (80, 80, 80), 1)
        cv2.putText(panel, "RECENT HISTORY:",
                    (10, 278), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (180, 180, 180), 1)
        for i, h in enumerate(history):
            color_val = max(int(160 - i * 25), 60)
            cv2.putText(panel, f"  > {h}",
                        (10, 298 + i * 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (color_val, color_val, color_val), 1)

        cv2.putText(panel, "Press Q or Esc to quit",
                    (10, 352), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (80, 80, 80), 1)

        dashboard = np.hstack([cam_display, panel])
        cv2.imshow("Real-Time Action Recognition", dashboard)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Dashboard closed.")

if __name__ == "__main__":
    run_dashboard()
import torch
import json
import numpy as np
from sklearn.metrics import classification_report
from data_loader import get_dataloaders
from r3d_model import get_r3d_model

DATASET_ROOT = "ucf101"
MODEL_PATH   = "best_model.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.json") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = get_r3d_model(len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)
print(f" Model loaded | Device: {DEVICE}")

_, val_loader, _ = get_dataloaders(DATASET_ROOT)
print(f" Val loader ready — {len(val_loader)} batches")

all_preds, all_labels = [], []
correct, total = 0, 0

print("\n Running evaluation...")
with torch.no_grad():
    for i, (clips, labels) in enumerate(val_loader):
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)
        outputs = model(clips)
        preds   = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(val_loader)} done...")

val_accuracy = correct / total * 100

print("\n" + "="*50)
print(f"  OVERALL VAL ACCURACY : {val_accuracy:.2f}%")
print(f"  Correct              : {correct}/{total}")
print("="*50)

target_names = [idx_to_class[i] for i in range(len(class_to_idx))]
report = classification_report(
    all_labels, all_preds,
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

class_f1 = {k: v["f1-score"] for k, v in report.items()
            if k not in ["accuracy", "macro avg", "weighted avg"]}
best5  = sorted(class_f1.items(), key=lambda x: x[1], reverse=True)[:5]
worst5 = sorted(class_f1.items(), key=lambda x: x[1])[:5]

print("\n BEST 5 CLASSES:")
for cls, score in best5:
    bar = "█" * int(score * 20)
    print(f"  {cls:30s} F1: {score:.2f}  {bar}")

print("\n  WORST 5 CLASSES:")
for cls, score in worst5:
    bar = "█" * int(score * 20)
    print(f"  {cls:30s} F1: {score:.2f}  {bar}")

print(f"\n  Macro Avg F1    : {report['macro avg']['f1-score']:.4f}")
print(f"  Weighted Avg F1 : {report['weighted avg']['f1-score']:.4f}")

results = {
    "val_accuracy"     : round(val_accuracy, 2),
    "correct"          : correct,   
    "total"            : total,
    "macro_avg_f1"     : round(report["macro avg"]["f1-score"], 4),
    "weighted_avg_f1"  : round(report["weighted avg"]["f1-score"], 4),
    "best_5_classes"   : best5,
    "worst_5_classes"  : worst5
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n Results saved to evaluation_results.json")
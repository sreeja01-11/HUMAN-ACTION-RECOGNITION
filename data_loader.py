import os
import pandas as pd
from torch.utils.data import DataLoader
from dataset import UCF101Dataset

def get_dataloaders(dataset_root, batch_size=8, num_workers=2):
    
    train_df = pd.read_csv(os.path.join(dataset_root, 'train.csv'))
    val_df   = pd.read_csv(os.path.join(dataset_root, 'val.csv'))

    all_classes  = sorted(train_df['label'].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    def build_lists(df):
        paths, labels = [], []
        for _, row in df.iterrows():
            rel  = row['clip_path'].lstrip('/')          
            full = os.path.join(dataset_root, rel)
            if os.path.exists(full):
                paths.append(full)
                labels.append(class_to_idx[row['label']])
        return paths, labels

    train_paths, train_labels = build_lists(train_df)
    val_paths,   val_labels   = build_lists(val_df)

    print(f"[data_loader] Classes : {len(class_to_idx)}")
    print(f"[data_loader] Train   : {len(train_paths)} videos")
    print(f"[data_loader] Val     : {len(val_paths)} videos")

    if len(train_paths) == 0:
        raise ValueError(
            "No training videos found!\n"
            "Check that dataset_root points to the UCF101 folder "
            "containing train.csv, val.csv, train/, val/ subfolders."
        )

    train_loader = DataLoader(
        UCF101Dataset(train_paths, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        UCF101Dataset(val_paths, val_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, class_to_idx
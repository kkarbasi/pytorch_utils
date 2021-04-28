import torch
from torch.utils.data import Dataset
import pandas as pd


class CSVDataset(Dataset):
    """
    A pytorch Dataset for CSV feature and target files

    Usage:

        training_data = CSVDataset(
            feature_path='/path/to/file/containing/features/as/columns.csv',
            annotation_file='/path/to/file/containing/labels/as/columns.csv',
            feature_cols=['feature 1', 'feature 2'],
            label_cols=['label'],
        )

        then:

            from torch.utils.data import DataLoader
            train_dataloader = DataLoader(training_data, batch_size=64,
                                          shuffle=True)
    """
    def __init__(self,
                 feature_path,
                 annotation_file,
                 feature_cols,
                 label_cols,
                 transform=None,
                 target_transform=None):
        self.X = pd.read_csv(feature_path)[feature_cols]
        self.y = pd.read_csv(annotation_file)[label_cols]
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        X = torch.from_numpy(self.X.iloc[[idx]].values)
        y = self.y.iloc[[idx]].values[0]
        if self.transform:
            X = self.transform(X)
        if self.target_transform:
            y = self.target_transform(y)
        return X, y

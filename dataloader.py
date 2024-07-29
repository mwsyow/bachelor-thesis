from utils import utility
from torch.utils import data

class TabularDataset(data.Dataset):
    def __init__(self, df, encoders, features, is_clf):
        assert encoders['ordinal'] != None and features['ordinal'] != None, "need to pass ordinal encoders and features"

        self.data = {}
        self.data['nominal'] = utility.concat(encoders['nominal'](df[features['nominal']])) if 'nominal' in encoders and 'nominal' in features else None
        self.data['ordinal'] = utility.concat(encoders['ordinal'](df[features['ordinal']]))
        self.data['continuous'] = df[features['continuous']].to_numpy() if 'continuous' in features else None
        self.data['label'] = utility.concat(encoders['label'](df[features['label']])) if is_clf else df[features['label']].to_numpy()
    def __len__(self):
        return len(self.data['label'])
    
    def __getitem__(self, idx):
        item = {}
        if not self.data['continuous'] is None:
            item['continuous'] = self.data['continuous'][idx]
        if not self.data['nominal'] is None:
            item['nominal'] = self.data['nominal'][idx] 
        item['ordinal'] = self.data['ordinal'][idx]
        item['label'] = self.data['label'][idx]
        return item
        
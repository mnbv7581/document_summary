# data loading
from datasets import load_dataset

class ArticleDataset:
    def __init__(self, data_files, data_type="json"):
        self.dataset = load_dataset(data_type, data_files=data_files)

    def create_datasets(self):

        # Split the dataset into training and validation datasets
        data = self.dataset['train'].train_test_split(train_size=0.8, seed=42)
        data['val'] = data['test']
        # explore data
        data.items()

        return data
        
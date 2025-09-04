import dotenv

dotenv.load_dotenv()
from datasets import load_dataset


def get_synth_dataset(lang: str, train=True, num=10000):
    if train:
        split = f"train_{lang.lower()}"
    else:
        split = f"test_{lang.lower()}"
    full_dataset = load_dataset("continuedev/instinct-data")
    dataset = full_dataset[split]
    
    # If num is greater than or equal to dataset size, return full dataset
    if num >= len(dataset):
        return dataset
    
    # Random sample num samples
    dataset = dataset.shuffle(seed=42)  # Optional: set seed for reproducibility
    dataset = dataset.select(range(num))
    
    return dataset

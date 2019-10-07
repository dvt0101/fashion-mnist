# Zalando's MNIST fashion replacement
This repo is archived.

Zalando recently released an MNIST replacement. The issues with using MNIST are
known but you can read about the dataset and their motivation [here](https://github.com/zalandoresearch/fashion-mnist).

### Training
```
python train.py

        --model         # specify model, (FashionSimpleNet, resnet18)
        --patience      # early stopping
        --batch_size    # batch size
        --nepochs       # max epochs
        --nworkers      # number of workers
        --seed          # random seed
        --data          # MNIST or FashionMNIST
```


### Results
|   | FashionSimpleNet | ResNest18 |
| ------------- | ------------- |-----------|
| MNIST  | 0.994  | 0.994|
| FashionMNIST  | 0.923  | 0.920|

### Improve by Mr.Thang-1612615
Augmentation: 
class Transformer Data:
    class DatasetTransformer(torch.utils.data.Dataset):
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)
    Augmentation:
        RandomRotation:
            train_transforms1 = transforms.Compose([
                transforms.RandomRotation(20)
            ])
        RandomCrop:
            train_transforms2 = transforms.Compose([
                transforms.RandomCrop(28, padding=4)
            ])
        RandomHorizotalFlip:
            train_transforms3 = transforms.Compose([
                transforms.RandomHorizontalFlip()
            ])
        Grayscal, Normalize:
            train_transforms4 = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])

####Results 
    |   | Training | Testing |
    | ------------- | ------------- |-----------|
    | FashionMNIST  | 0.9907  | 0.9469|

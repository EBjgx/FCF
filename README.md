# Fusion of label correction and filtering
AAAI'24: Which is More Effective in Label Noise Cleaning, Correction or Filtering?(Official Pytorch implementation for noisy labels).  


This is the code for the paper: Which is More Effective in Label Noise Cleaning, Correction or Filtering? 




## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 2.0.1 
- Torchvision 0.15.2


## Running our method on benchmark datasets (CIFAR-10 and CIFAR-100).
Here is an example:
```bash
python main.py --dataset cifar10 --corruption_type SymNoise(AsymNoise) --corruption_prob 0.6
```

The default network structure is Resnet34


## FCF Core Package Components
1.  **FCF/main.py** - Estimates the noise rates and error bounds of FCF.Start correcting during the first cleaning, and filtering during the second cleaning.
2.  **FCF/dataloader.py** - Load and process the dataset. 
3.  **FCF/resnet.py** - Create ResNet network as the backbone.




## Acknowledgements
We thank the Pytorch implementation on glc(https://github.com/mmazeika/glc) and learning-to-reweight-examples(https://github.com/danieltan07/learning-to-reweight-examples).

# FCF: Fusion of Label Correction and Filtering
AAAI'24: Which is More Effective in Label Noise Cleaning, Correction or Filtering? (Official Pytorch implementation for noisy labels).  

This is the code for the paper: Which is More Effective in Label Noise Cleaning, Correction or Filtering? 


## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 2.0.1 
- Torchvision 0.15.2


## Running Our Method on Benchmark Datasets (CIFAR-10 and CIFAR-100).
Run ` main.py` with the commands like below:
```bash
python main.py --dataset cifar10 --corruption_type SymNoise(AsymNoise) --corruption_prob 0.6
```

The default network structure is ResNet34.


## FCF Core Package Components
1.  **FCF/main.py** - Estimates the noise rates and error bounds of FCF. Start correcting during the first round of cleaning, and filtering during the second round of cleaning.
2.  **FCF/dataloader.py** - Load and process the dataset. 
3.  **FCF/resnet.py** - Create ResNet network as the backbone.



## Reference Codes
We refer to some official implementation codes

 - https://github.com/JackYFL/DISC
 - https://github.com/xjtushujun/CMW-Net
 - https://github.com/Kthyeon/FINE_official
 - https://github.com/WuYichen-97/
 - https://github.com/bhanML/Co-teaching
 - https://github.com/LiJunnan1992/DivideMix


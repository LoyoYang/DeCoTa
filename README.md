## MiCo: Mixup Co-training for Semi-supervised Domain Adaptation

### This is a Private Repository. Without consulting the owner, please do not distribute or publish.

### Requirements
The code is developed under Python 3.6.5 and PyTorch 1.4.0

Requirements are listed in requirements.txt

### Prepare Dataset
To reproduce the DomainNet results, download DomainNet from http://ai.bu.edu/M3SDA/ following the instructions on the page.

Your dataset root is expected to contain folders named after all the domains, for example: 

```PATH/TO/DATASET/ROOT/clipart```

### Train your own MiCo
There are 7 adaptation scenraios on DomainNet experiment. Specify the Source and Target domain by either --source XXX --target
YYY or 
python main.py --root PATH/TO/DATASET/ROOT/ --st SETTING_INDEX

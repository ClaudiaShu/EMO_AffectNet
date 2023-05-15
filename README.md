# EMO_AffectNet

This is the evaluation code for our Paper "Revisiting Self-Supervised Contrastive Learning for Facial Expression Recognition". For the main repository, please refer to [ClaudiaShu/SSL-FER](https://github.com/ClaudiaShu/SSL-FER).

## Code overview

```
.    
├── models 			
│   ├── model_MLP.py                    # MLP layer              
│   └── model_RES.py                    # backbone ResNet model      
├── LFW.py                              # main program for LFW evaluation
├── AffectNet_create_csv.py             # process dataset into csv
├── AffectNet_prep.py                   # Affectnet preparation
├── file2hdf5.py                        # convert csv to hdf5
├── data_aug.py                         # applied image augmentation
├── data_loader.py                      # dataloader for AffectNet
└── main.py                             # main program for Affectnet evaluation
```

To evaluate with LFW, please run:
```commandline
python LFW.py 
```

To evaluate with AffectNet, please run:
```commandline
python main.py 
```

For AffectNet, we support two different kinds of data inputs - csv and hdf5. Please convert the csv files with `file2hdf5.py` and uncomment below script in `main.py` if you want to train with hdf5. 

```python
train_data = "/data2/yshu21/data/AffectNet/cache/AffectNet_train.hdf5"
valid_data = "/data2/yshu21/data/AffectNet/cache/AffectNet_valid.hdf5"
train_dataset = AFFECTNET_hdf5(train_data, transform=train_transform, task=args.task)
valid_dataset = AFFECTNET_hdf5(valid_data, transform=valid_transform, task=args.task)
```
The AffectNet dataset can be accessed upon successful application through [the official webpage](http://mohammadmahoor.com/affectnet/).

# EMO_AffectNet

This is the evaluation code for our Paper "Revisiting Self-Supervised Contrastive Learning for Facial Expression Recognition".

## Code overview

```
.    
├── models 			
│   ├── model_MLP.py                    # MLP layer              
│   └── model_RES.py                    # backbone ResNet model      
├── LFW.py                              # main program for LFW evaluation
├── AffectNet_create_csv.py.py          # preprocess dataset
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

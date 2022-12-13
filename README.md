# STAR-RL

This is the implementation of STAR-RL

## Environment

Python 3.6.1 <br/>
Pytorch 0.3.1.post2 <br/>
torchvision 0.2.0 <br/>
numpy 1.14.2 <br/>
tensorboardX 1.7 <br/>

## Training
1. We first train the spatial manager (spM) and patch work (PW), run
```
sh ./launch/train_spM_PW.sh
```

2. Then, we train the spatial manager (spM), temporal manager (tpM) and patch work (PW) together, run
```
sh ./launch/train_all.sh
```

## Testing

Run
```
sh ./launch/test.sh
```

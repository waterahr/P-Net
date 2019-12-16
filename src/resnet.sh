epoch=$1
gpu=$2
python train.py -m ResNet50 -d PETA -e $epoch -g $gpu -c 61 -b 16 -s adam
python test.py -m ResNet50 -d PETA -s thr3 -g $gpu -w adam_
python train.py -m ResNet50 -d RAP -e $epoch -g $gpu -c 51 -b 16 -s adam
python test.py -m ResNet50 -d RAP -s thr3 -g $gpu -w adam_
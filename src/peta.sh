epoch=$1
gpu=$2
#python train.py -m Inception -d PETA -g $gpu -e $epoch -c 61 -b 64
#python train.py -m PartPoolInception -d PETA -e $epoch -s v1 -g $gpu -c 61 -b 32
#python train.py -m PartPoolInception -d PETA -e $epoch -s v2 -g $gpu -c 61 -b 32
#python train.py -m PartPoolInception -d PETA -e $epoch -s v3 -g $gpu -c 61 -b 32
#python train.py -m PartPoolInception -d PETA -e $epoch -s v4 -g $gpu -c 61 -b 32
#python train.py -m PartPoolInception -d PETA -e $epoch -s v5 -g $gpu -c 61 -b 32
### python train.py -m PartPoolInception -d PETA -e $epoch -s v6 -g $gpu -c 61 -b 1
#python train.py -m PartPoolInception -d PETA -e $epoch -s triplet_v7 -g $gpu -c 61 -b 32
python train.py -m PartPoolInception -d PETA -e $epoch -s center_v7 -g $gpu -c 61 -b 32 -w ../models/PETA/PartPoolInception/center_v7_final200iter_model.h5
#python train.py -m InceptionV3 -d PETA -g $gpu -e $epoch -c 61 -b 32
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v1 -c 61 -b 32 -g $gpu 
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v2 -c 61 -b 32 -g $gpu
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v3 -c 61 -b 32 -g $gpu 
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v4 -c 61 -b 32 -g $gpu 
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v5 -c 61 -b 1 -g $gpu
#python train.py -m ResNet50 -d PETA -g $gpu -e $epoch -c 61 -b 16
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -s v1 -c 61 -b 16 -g $gpu
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -s v2 -c 61 -b 16 -g $gpu
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -s v3 -c 61 -b 16 -g $gpu
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -s v4 -c 61 -b 16 -g $gpu
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -s v5 -c 61 -b 16 -g $gpu
#sh runtest.sh
#python test.py -m Inception -d PETA -g $gpu -c 61
#python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -s thr3 -w triplet_v7_
python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -s thr3 -w center_v7_
### python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -w v6_
#python test.py -m InceptionV3 -d PETA -g $gpu -c 61
#python test.py -m PartPoolInceptionV3 -d PETA -g $gpu -c 61 -w v1_
#python test.py -m PartPoolInceptionV3 -d PETA -g $gpu -c 61 -w v4_
#python test.py -m PartPoolInceptionV3 -d PETA -g $gpu -c 61 -w v5_
#python test.py -m ResNet50 -d PETA -g $gpu -c 61
#python test.py -m PartPoolResNet50 -d PETA -g $gpu -c 61 -w v1_
#python test.py -m PartPoolResNet50 -d PETA -g $gpu -c 61 -w v2_
#python test.py -m PartPoolResNet50 -d PETA -g $gpu -c 61 -w v3_
#python test.py -m PartPoolResNet50 -d PETA -g $gpu -c 61 -w v4_
#python test.py -m PartPoolResNet50 -d PETA -g $gpu -c 61 -w v5_
epoch=$1
gpu=$2
#python train.py -m Inception -d PETA -g $gpu -e $epoch -c 61 -b 64 -s coarse
#python train.py -m PartPoolInception -d PETA -e $epoch -g $gpu -c 61 -b 32 -s coarse_v1
#python train.py -m PartPoolInception -d PETA -e $epoch -g $gpu -c 61 -b 32 -s coarse_v2
#python train.py -m PartPoolInception -d PETA -e $epoch -g $gpu -c 61 -b 32 -s coarse_v3
#python train.py -m PartPoolInception -d PETA -e $epoch -g $gpu -c 61 -b 32 -s coarse_v4
#python train.py -m PartPoolInception -d PETA -e $epoch -s v5 -g $gpu -c 61 -b 32 -s coarse
#python train.py -m InceptionV3 -d PETA -g $gpu -e $epoch -c 61 -b 32 -s coarse
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -c 61 -b 32 -g $gpu -s coarse_v1
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -c 61 -b 32 -g $gpu -s coarse_v2
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -c 61 -b 32 -g $gpu -s coarse_v3
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -c 61 -b 32 -g $gpu -s coarse_v4
#python train.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v5 -c 61 -b 1 -g $gpu -s coarse
#python train.py -m ResNet50 -d PETA -g $gpu -e $epoch -c 61 -b 16 -s coarse
python train.py -m PartPoolResNet50 -d PETA -e $epoch -c 61 -b 16 -g $gpu -s coarse_v1
python train.py -m PartPoolResNet50 -d PETA -e $epoch -c 61 -b 16 -g $gpu -s coarse_v2
python train.py -m PartPoolResNet50 -d PETA -e $epoch -c 61 -b 16 -g $gpu -s coarse_v3
python train.py -m PartPoolResNet50 -d PETA -e $epoch -c 61 -b 16 -g $gpu -s coarse_v4
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -c 61 -b 16 -g $gpu -s coarse_v5
#python test.py -m PartPoolResNet50 -d PETA -e $epoch -s v5 -c 61 -b 16 -g $gpu -s coarse
python test.py -m Inception -d PETA -g $gpu -c 61 -s coarse -w coarse_
python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -s coarse_v1 -w coarse_v1_
python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -s coarse_v2 -w coarse_v2_
python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -s coarse_v3 -w coarse_v3_
python test.py -m PartPoolInception -d PETA -g $gpu -c 61 -s coarse_v4 -w coarse_v4_
#python test.py -m PartPoolInception -d PETA -e $epoch -s v5 -g $gpu -c 61 -b 32 -s coarse
python test.py -m InceptionV3 -d PETA -g $gpu -c 61 -s coarse -w coarse_
python test.py -m PartPoolInceptionV3 -d PETA -c 61 -g $gpu -s coarse_v1 -w coarse_v1_
python test.py -m PartPoolInceptionV3 -d PETA -c 61 -g $gpu -s coarse_v2 -w coarse_v2_
python test.py -m PartPoolInceptionV3 -d PETA -c 61 -g $gpu -s coarse_v3 -w coarse_v3_
python test.py -m PartPoolInceptionV3 -d PETA -c 61 -g $gpu -s coarse_v4 -w coarse_v4_
#python test.py -m PartPoolInceptionV3 -d PETA -e $epoch -s v5 -c 61 -b 1 -g $gpu -s coarse
python test.py -m ResNet50 -d PETA -g $gpu -c 61 -s coarse -w coarse_
python test.py -m PartPoolResNet50 -d PETA -c 61 -g $gpu -s coarse_v1 -w coarse_v1_
python test.py -m PartPoolResNet50 -d PETA -c 61 -g $gpu -s coarse_v2 -w coarse_v2_
python test.py -m PartPoolResNet50 -d PETA -c 61 -g $gpu -s coarse_v3 -w coarse_v3_
python test.py -m PartPoolResNet50 -d PETA -c 61 -g $gpu -s coarse_v4 -w coarse_v4_
#python test.py -m PartPoolResNet50 -d PETA -e $epoch -c 61 -b 16 -g $gpu -s coarse_v5
#python train.py -m PartPoolResNet50 -d PETA -e $epoch -s v5 -c 61 -b 16 -g $gpu -s coarse
epoch=$1
gpu=$2
python train.py -m Inception -d PA100K -g $gpu -e $epoch -c 26 -b 64
python train.py -m PartPoolInception -d PA100K -e $epoch -s v1 -g $gpu -c 26 -b 32
python train.py -m PartPoolInception -d PA100K -e $epoch -s v2 -g $gpu -c 26 -b 32
python train.py -m PartPoolInception -d PA100K -e $epoch -s v3 -g $gpu -c 26 -b 32
python train.py -m PartPoolInception -d PA100K -e $epoch -s v4 -g $gpu -c 26 -b 32
python train.py -m PartPoolInception -d PA100K -e $epoch -s v5 -g $gpu -c 26 -b 32
### python train.py -m PartPoolInception -d PA100K -e $epoch -s v6 -g $gpu -c 26 -b 1
python train.py -m InceptionV3 -d PA100K -g $gpu -e $epoch -c 26 -b 32
python train.py -m PartPoolInceptionV3 -d PA100K -e $epoch -s v1 -c 26 -b 32 -g $gpu 
python train.py -m PartPoolInceptionV3 -d PA100K -e $epoch -s v2 -c 26 -b 32 -g $gpu
python train.py -m PartPoolInceptionV3 -d PA100K -e $epoch -s v3 -c 26 -b 32 -g $gpu 
python train.py -m PartPoolInceptionV3 -d PA100K -e $epoch -s v4 -c 26 -b 32 -g $gpu 
python train.py -m PartPoolInceptionV3 -d PA100K -e $epoch -s v5 -c 26 -b 1 -g $gpu
python train.py -m ResNet50 -d PA100K -g $gpu -e $epoch -c 26 -b 16
python train.py -m PartPoolResNet50 -d PA100K -e $epoch -s v1 -c 26 -b 16 -g $gpu
python train.py -m PartPoolResNet50 -d PA100K -e $epoch -s v2 -c 26 -b 16 -g $gpu
python train.py -m PartPoolResNet50 -d PA100K -e $epoch -s v3 -c 26 -b 16 -g $gpu
python train.py -m PartPoolResNet50 -d PA100K -e $epoch -s v4 -c 26 -b 16 -g $gpu
python train.py -m PartPoolResNet50 -d PA100K -e $epoch -s v5 -c 26 -b 16 -g $gpu
#sh runtest.sh
python test.py -m Inception -d PA100K -g $gpu -c 26
python test.py -m PartPoolInception -d PA100K -g $gpu -c 26 -w v1_
python test.py -m PartPoolInception -d PA100K -g $gpu -c 26 -w v2_
python test.py -m PartPoolInception -d PA100K -g $gpu -c 26 -w v3_
python test.py -m PartPoolInception -d PA100K -g $gpu -c 26 -w v4_
python test.py -m PartPoolInception -d PA100K -g $gpu -c 26 -w v5_
#python test.py -m PartPoolInception -d PA100K -g $gpu -c 26 -w v6_
python test.py -m InceptionV3 -d PA100K -g $gpu -c 26
python test.py -m PartPoolInceptionV3 -d PA100K -g $gpu -c 26 -w v1_
python test.py -m PartPoolInceptionV3 -d PA100K -g $gpu -c 26 -w v2_
python test.py -m PartPoolInceptionV3 -d PA100K -g $gpu -c 26 -w v3_
python test.py -m PartPoolInceptionV3 -d PA100K -g $gpu -c 26 -w v4_
python test.py -m PartPoolInceptionV3 -d PA100K -g $gpu -c 26 -w v5_
python test.py -m ResNet50 -d PA100K -g $gpu -c 26
python test.py -m PartPoolResNet50 -d PA100K -g $gpu -c 26 -w v1_
python test.py -m PartPoolResNet50 -d PA100K -g $gpu -c 26 -w v2_
python test.py -m PartPoolResNet50 -d PA100K -g $gpu -c 26 -w v3_
python test.py -m PartPoolResNet50 -d PA100K -g $gpu -c 26 -w v4_
python test.py -m PartPoolResNet50 -d PA100K -g $gpu -c 26 -w v5_
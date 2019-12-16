epoch=$1
gpu=$2
#name=$3
#python test.py -m Inception -d RAP
#python train.py -m PartPoolInception -d RAP -e $epoch -s ${name}_v1 -g $gpu
#python test.py -m PartPoolInception -d RAP -w ${name}_v1_ -g $gpu
#python test.py -m InceptionV3 -d RAP
#python test.py -m PartPoolInceptionV3 -d RAP -w v5_ -g 2
#python test.py -m ResNet50 -d RAP
#python test.py -m PartPoolResNet50 -d RAP -w v5_ -g 2

python train.py -m PartPoolInception -d RAP -e $epoch -s v8 -g $gpu -w 
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v8_
#python train.py -m PartPoolInception -d PETA -c 61 -e $epoch -s v8 -g $gpu
#python test.py -m PartPoolInception -d PETA -c 61 -s thr3 -g $gpu -w v8_
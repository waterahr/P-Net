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

python train.py -m PartPoolInception -d RAP -e $epoch -s v1 -g $gpu -i 918 -w ../models/RAP/PartPoolInception/v1_sgd_epoch918_valloss0.858514.hdf5
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v1_
#python train.py -m DenseNet201 -b 16 -d RAP -e $epoch -g $gpu
#python test.py -m DenseNet201 -d RAP -s thr3 -g $gpu 
#python train.py -m ResNet50 -d RAP -e $epoch -g $gpu
#python test.py -m ResNet50 -d RAP -s thr3 -g $gpu 

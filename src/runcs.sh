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

python train.py -m PartPoolInception -d RAP -e $epoch -s v8 -g $gpu -w ../models/RAP/PartPoolInception/v8_sgd_epoch163_valloss0.859688.hdf5 -i 163
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v8_
python train.py -m PartPoolInception -d RAP -e $epoch -s center_v8 -g $gpu -w ../models/RAP/PartPoolInception/center_v8_epoch042_valloss0.853647.hdf5
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v8_
python train.py -m PartPoolInception -d RAP -e $epoch -s v9 -g $gpu -w ../models/RAP/PartPoolInception/v9_epoch497_valloss0.698967.hdf5
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v9_
python train.py -m PartPoolInception -d RAP -e $epoch -s center_v9 -g $gpu -w ../models/RAP/PartPoolInception/center_v9_epoch497_valloss0.779923.hdf5
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v9_
python train.py -m PartPoolInception -d RAP -e $epoch -s v10 -g $gpu -w ../models/RAP/PartPoolInception/v10_epoch077_valloss0.852263.hdf5
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v10_
python train.py -m PartPoolInception -d RAP -e $epoch -s center_v10 -g $gpu -w ../models/RAP/PartPoolInception/center_v10_epoch395_valloss0.850809.hdf5
python test.py -m PartPoolInception -d RAP -s thr3 -g $gpu -w v10_
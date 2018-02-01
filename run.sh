#!/bin/bash


if [ "$1" == "train" ]
then
    python src/train.py train
elif [ "$1" == "predict" ]
then
    python src/train.py test
else
    python src/train.py train
    python src/train.py test
fi

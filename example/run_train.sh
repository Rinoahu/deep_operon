#!/bin/bash

tstime='/usr/bin/time -v'
#python deep_operon.py train all_in_one.fasta training_deep.txt

pbzip2 -d all_in_one.fasta.bz2
python ../deep_operon.py train all_in_one.fasta training_deep.txt 2d
pbzip2 --best all_in_one.fasta


#python deep_operon.py predict all_in_one.fasta training_deep.txt training_deep.txt.lgb > training_deep.txt.lgb.predict_test2
#exit 0

#for i in *.hdf5
#do
#    echo $i
#    python ../deep_operon.py predict all_in_one.fasta training_deep.txt $i > $i.predict_test2
#    #mv $i ./models
#done

#python deep_operon.py predict all_in_one.fasta training_deep.txt training_deep.txt_2d > training_deep.txt.predict_test2

#python deep_operon.py predict all_in_one.fasta training_deep.txt model3.hdf5 > training_deep.txt.predict3

#python deep_operon.py predict all_in_one.fasta training_deep.txt model3_256.hdf5 > training_deep.txt.predict3_256

#python deep_operon.py predict all_in_one.fasta training_deep.txt model4.hdf5 > training_deep.txt.predict4

#python deep_operon.py predict all_in_one.fasta training_deep.txt model5_256.hdf5 > training_deep.txt.predict5_256



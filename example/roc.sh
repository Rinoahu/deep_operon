#!/bin/bash
#qry=$1

#for i in `ls ./*.predict*`
for i in ./*.predict*
do
    echo $i
    awk '{if($11=="True"&&$12=="True") tp+=1; else if ($11=="False"&&$12=="True") fp+=1; else if ($11=="True" && $12=="False") fn+=1} END {prec=tp/(tp+fp); rec=tp/(tp+fn); f1=2*prec*rec/(prec+rec); print " Precision: " prec " Recall: " rec " F1: " f1}' $i

done

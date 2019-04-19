#!/bin/sh
for data in 2
do
  for gan in 1 11 2
  do
    for model in 1
    do
      for i in 0.001 0.005 0.007 0.01 0.015
      do
        python run_ttest.py $data $gan $i $model;
      done
    done
  done
done

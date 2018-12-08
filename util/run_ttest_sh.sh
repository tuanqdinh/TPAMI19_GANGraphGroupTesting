#!/bin/sh
for data in 1 2
do
  for gan in 1 11 2 21
  do
    for model in 1 2 3
    do
      for i in 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
      do
        python run_ttest.py $data $gan $model $i;
      done
    done
  done
done

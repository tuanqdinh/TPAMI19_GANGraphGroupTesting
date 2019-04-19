#!/bin/sh
for data in 2
do
  for gan in 1 11 2
  do
    for model in 1
    do
      for cn in 1 2
      do
        for i in 0.001 0.005 0.007 0.01 0.015
        do
          python main_wgan.py --off_data $data --off_ctrl $cn --off_gan $gan --off_model $model --alpha $i
        done
      done
    done
  done
done

#0.1 0.11 0.13 0.15 0.2 0.3 0.5 0.7 1 10

#!/bin/sh
for data in 2
do
  for gan in 1 11 2 21
  do
    for model in 1 2
    do
      for cn in 1 2
      do
        for i in 0.005 0.0.07 0.09 0
        do
          python main_wgan.py --off_data $data --off_ctrl $cn --off_gan $gan --off_model $model --alpha $i
        done
      done
    done
  done
done

#!/bin/sh
for data in 1
do
  for gan in 1 11 2 21
  do
    for model in 1 2
    do
      for cn in 1 2
      do
        for i in 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09
        do
          python gan_mesh.py --off_data $data --off_ctrl $cn --off_gan $gan --off_model $model --alpha $i
        done
      done
    done
  done
done

#!/bin/bash
for r in 2 4 8 16 32
do
  for alpha in 0.6 0.2 0.06 0.02
  do
    for lambd in 0.01 0.04 0.1 0.4
    do
      python sgd.py --r $r --alpha $alpha --lambd $lambd
    done
  done
done
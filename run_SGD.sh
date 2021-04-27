#!/bin/bash

# python3 project.py --train train_ratings.csv --test test_ratings.csv --alg SGD --result res.txt
python3 -m debugpy --listen 5678 --wait-for-client project.py --train train_ratings.csv --test test_ratings.csv --alg SGD --result res.txt
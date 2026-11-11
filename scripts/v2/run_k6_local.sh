#!/bin/bash
export AURUM_V2F=/home/z/my-project/kaggle_work/v2/k1_out4
export AURUM_DS=/home/z/my-project/kaggle_work/aurum/dataset_p2
export AURUM_K2DIR=/home/z/my-project/kaggle_work/v2/k2_out2
export AURUM_K3DIR=/home/z/my-project/kaggle_work/v2/k3_out2
export AURUM_V13DIR=/home/z/my-project/kaggle_work/v2/v13_art
export AURUM_K6OUT=/home/z/my-project/kaggle_work/v2/k6_out1
cd /home/z/my-project
python3 aurum_v2/kernels/k6_backtest.py

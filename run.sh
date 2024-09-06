#!/bin/bash
python main.py --operator integer --alphas "[0.9, 1.1]" --model 'fc1' &
python main.py --operator integer --alphas "[0.9, 1.1]" --model 'resnet18' &
python main.py --operator fractional --alphas "[0.9, 1.1]" --model 'fc1' &
python main.py --operator fractional --alphas "[0.9, 1.1]" --model 'resnet18' &
python main.py --operator multi_fractional --alphas "[0.9, 1.1]" --model 'fc1' &
python main.py --operator multi_fractional --alphas "[0.9, 1.1]" --model 'resnet18' &
python main.py --operator distributed_fractional --alphas "[0.9, 1.1]" --model 'fc1' &
python main.py --operator distributed_fractional --alphas "[0.9, 1.1]" --model 'resnet18' &

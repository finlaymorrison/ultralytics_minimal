#!/bin/python3

from ultralytics import YOLO
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

DATASET_DIR = "datasets" 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help='Path to dataset.')
parser.add_argument('-v', '--variant', choices=["n", "s", "m", "l", "x"], type=str, required=True, help='YOLO Model variant to use.')
parser.add_argument('-e', '--epochs', type=int, default='10', help='Number of training epochs.')
parser.add_argument('-b', '--batch', type=int, default='16', help='Number of samples per batch.')
args = parser.parse_args()

def main(args):
    model = YOLO(f"{DATASET_DIR}/{args.data}/yolov8{args.variant}.yaml")
    model.train(data=f"{DATASET_DIR}/{args.data}/dataset.yaml", epochs=args.epochs, batch=args.batch, name=f"{args.data}-{args.variant}-{args.epochs}-{args.batch}")

if __name__ == "__main__":
    main(args)
#!/bin/python
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
            description='Provide input data file')
    parser.add_argument('input_data_file', type=str, help='input data file')
    parser.add_argument('graphs_folder', type=str, help='graphs folder')
    parser.add_argument('results_folder', type=str, help='results folder')
    return parser.parse_args()

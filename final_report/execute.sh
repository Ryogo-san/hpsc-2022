#!/bin/bash
set -ue

nvcc main.cu -std=c++11 -Xcompiler -fopenmp
./a.out

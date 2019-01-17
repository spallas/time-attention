#!/usr/bin/env bash

cd data

# Download SML2010 dataset from UCI
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip
unzip NEW-DATA.zip

# Remove first 4 chars of each file making it a proper csv
tail -c +4 NEW-DATA-1.T15.txt > NEW-DATA-1.T15.csv
tail -c +4 NEW-DATA-2.T15.txt > NEW-DATA-2.T15.csv



# Clean
rm NEW-DATA.zip
rm NEW-DATA-1.T15.txt
rm NEW-DATA-2.T15.txt
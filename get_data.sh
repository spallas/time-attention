#!/usr/bin/env bash

if [ ! -d data ]; then
    mkdir data
fi

# Download NASDAQ100 dataset from author's
if [ ! -f data/nasdaq100.zip ]; then
    wget http://cseweb.ucsd.edu/~yaq007/nasdaq100.zip
    mv nasdaq100.zip data
fi

# Download SML2010 dataset from UCI
if [ ! -f data/SML2010.zip ]; then
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip
    mv NEW-DATA.zip data/SML2010.zip
fi

if [ -d  data/SML2010 ]; then
    rm -r  data/SML2010
fi

if [ -d data/nasdaq100 ]; then
    rm -r data/nasdaq100
fi

# SML2010 Processing
unzip -q data/SML2010.zip -d data/SML2010

# Remove first 4 chars of each file making it a proper csv
tail -c +4 data/SML2010/NEW-DATA-1.T15.txt > data/SML2010/NEW-DATA-1.T15.csv
tail -c +4 data/SML2010/NEW-DATA-2.T15.txt > data/SML2010/NEW-DATA-2.T15.csv

# Clean
rm data/SML2010/NEW-DATA-1.T15.txt
rm data/SML2010/NEW-DATA-2.T15.txt

# NASDAQ100 Processing
unzip -q data/nasdaq100.zip -d data
rm -r data/__MACOSX

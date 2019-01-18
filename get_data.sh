#!/usr/bin/env bash

if [ ! -d data ]; then
    mkdir data
fi

cd data

# Download NASDAQ100 dataset from author's
if [ ! -f nasdaq100.zip ]; then
    wget http://cseweb.ucsd.edu/~yaq007/nasdaq100.zip
fi

# Download SML2010 dataset from UCI
if [ ! -f NEW-DATA.zip ]; then
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00274/NEW-DATA.zip
fi

if [ -d SML2010 ]; then
    rm -r SML2010
fi

if [ -d nasdaq100 ]; then
    rm -r nasdaq100
fi

# SML2010 Processing
mv NEW-DATA.zip SML2010.zip
unzip SML2010.zip -q -d SML2010
cd SML2010

# Remove first 4 chars of each file making it a proper csv
tail -c +4 NEW-DATA-1.T15.txt > NEW-DATA-1.T15.csv
tail -c +4 NEW-DATA-2.T15.txt > NEW-DATA-2.T15.csv

# Clean
rm NEW-DATA-1.T15.txt
rm NEW-DATA-2.T15.txt

cd ..

# NASDAQ100 Processing
unzip -q qnasdaq100.zip
rm -r __MACOSX

#!/bin/bash

cd lab"$1"
pdflatex lab"$1".tex
rm *.aux *.log
cd ..

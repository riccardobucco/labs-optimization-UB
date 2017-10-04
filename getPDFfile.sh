#!/bin/bash

cd lab"$1"/TeX
pdflatex lab"$1".tex
rm *.aux *.log
mv lab"$1".pdf ..
cd ../..

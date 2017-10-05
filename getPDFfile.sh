#!/bin/bash

cd lab"$1"/TeX
pdflatex lab"$1".tex
pdflatex lab"$1".tex
rm *.aux *.log *.out
mv lab"$1".pdf ..
cd ../..

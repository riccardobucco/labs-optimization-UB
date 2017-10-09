#!/bin/bash

cd lab"$1"/Scripts
for script in *.py
do
    python "$script"
done
cd ../TeX
pdflatex lab"$1".tex
pdflatex lab"$1".tex
rm *.aux *.log *.out
mv lab"$1".pdf ..
cd ../..
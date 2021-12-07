# SensitivityAnalysis
## Overview
It performs PCA and clustering operations and calculates the lost variance.

## Requirements
* numpy
* matplotlib
* scipy
* sklearn
* pandas

## Installation
* pip3 install -r requirements.txt

## How To Use
* You have to specify the path of your file within the code
* Make sure you don't read any non-number columns, and eliminate constant columns
* You can choose the columns to ignore by specifying their names in the DataFrame.drop () inside the readData () function

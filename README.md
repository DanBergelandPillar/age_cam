# Age Camera

This group project by Pillar Technology artisans is being used to promote deep learning skills. 
The goal of this project is to create an app that uses a laptop camera and estimates the age of the person.

# Install and Setup
If using anaconda, use the condaEnvironment.txt file by running : conda create --name <env> --file condaEnvironment.txt

Else, to get the Python 3.7 pip libs, install Python 3.7 then run pip install -r requirements.txt

# Testing
From the root directory, run nosetests, or python -m unittest discover -s test/ -p '*_test.py'

# Data
This model is trained on the UTKFace data 'Aligned&Cropped Faces' dataset.
https://susanqq.github.io/UTKFace/

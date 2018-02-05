# Toxic Comment Classification Challenge

This project is the code for Kaggle [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and completion of Udacity Machine Learning Engineer capstone project.

This script can achieve 0.95 ROC AUC using single classifier.

# Run Script

To install required libraries:

`pip install requirement.txt` 

Download competition's data. The links are here: [Competition data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

There's no need to extract files from archives.

To train the model:

`./run.sh train`

To generate new submission:

`./run.sh predict`

To train and generate submission sequentially run:

`./run.sh`

You can change model parameter using `src/config.py`.




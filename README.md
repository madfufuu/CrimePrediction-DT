# Distributed Decision Tree with PySpark
Final project for CS6350 - distributed decision trees with PySpark

Team Members:
- Justin Dula
- Zixuan Yang
- Adharsh Rajendran
- Yuncheng Gao


Files Included:
- README.md - Contains general information and instructions on how to run the code.
- Group Project Report - Project report.
- DecisionTree.html - HTML version of the python notebook.
- DecisionTree.py - Source code of the project.
- output - Contains the final prediction of the decisiontree on the test dataset.

Build Environment:
- PySpark
- Python 3.8.x

## Dataset
Source:
- https://www.kaggle.com/c/sf-crime/overview
- https://www.kaggle.com/c/titanic

Hosted on S3:
- https://decisiontree-bigdata.s3.amazonaws.com/Titanic_test.csv
- https://decisiontree-bigdata.s3.amazonaws.com/Titanic_train.csv


Instructions to Run:
- The content of the DecisionTree.py file needs to be copied over and executed on a hosted AWS EMR Notebook.
- First initialize an AWS EMR cluster instance with the following configurations.
- Release: 
	- emr-5.32.0
- Software Configuration:
	- Hadoop 2.10.1, JupyterHub 1.1.0, Hive 2.3.7, JupyterEnterpriseGateway 2.1.0, Hue 4.8.0, Spark 2.4.7, Livy 0.7.0, Pig 0.17.0

Then an EMR Notebook instance could be initialized on the cluster we've just started.
On the EMR Notebook, select PySpark as the environment of execution and copypaste in the code from DecisionTree.py file block by block for execution.

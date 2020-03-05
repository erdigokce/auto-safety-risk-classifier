Auto Risk Estimator - Classifier
===============

Classifies the given auto data to related risk class with using Naive Bayes machine learning model.

Prerequisites
------------------

Make sure *virtualenv* is installed ::

    >>> pip install virtualenv

Setup
------------------

In order to setup environment simply run following in the project directory ::

    >>> setup.bat
    
This will create a virtual python environment in project directory and download all the dependencies there. 

Configuration
------------------

Following pairs should be configured as desired in the *application/config/__init__.py* ::

	THRESHOLD_OPTIMIZER=0,  EVALUATION_METHOD=dict(METHOD_NAME='SplitByRatio', METHOD_VALUE=0.66)
	
	THRESHOLD_OPTIMIZER=1,  EVALUATION_METHOD=dict(METHOD_NAME='KFold', METHOD_VALUE=10)

Run
------------------

To run application, simply run the following in project directory ::

    >>> run.bat

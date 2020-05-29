
We will use the Python programming language for all assignments in this course. Specifically, we will use a few popular libraries (numpy, matplotlib, math) for scientific computing.

We expect that many of you already have some basic experience with Python and Numpy. We also provide basic guide to learn Python and Numpy.


Setup
-----
Download and install Anaconda with Python3.6 version:
- Download at the website: https://www.anaconda.com/download/
- Install Python3.6 version(not Python 2.7)
Anaconda will include all the Python libraries we need. 

Start programming:
Open Anaconda and choose Spyder to start your programming exercise.


Python & Numpy Tutorial
-----------------------
- Official Python tutorial: https://docs.python.org/3/tutorial/
- Official Numpy tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
- Good tutorial sources: http://cs231n.github.io/python-numpy-tutorial/ 


Dataset Descriptions
--------------------
We will use part of MNIST image dataset for all the assignments. All the data are in 'data' folder namely 'train.txt' and 'test.txt'. Here are the details of the dataset information:
        		     samples   image size   labels 
Training Dataset      1561      16*16      1 or 5
Testing Dataset       424       16*16      1 or 5

We already extracted two features discussed in class, so you can directly use these features as your input. 
feature1: symmetry
feature2: average intensity
Here are the details of the feature information:
		          samples   feature numbers   labels 
Training Dataset     1561           2           1 or 5
Testing Dataset       424           2           1 or 5







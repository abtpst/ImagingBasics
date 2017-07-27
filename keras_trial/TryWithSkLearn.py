'''
Created on Jul 27, 2017

@author: abhijit.tomar
'''
import seaborn as sns
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

if __name__ == '__main__':
    
    iris = pd.read_csv('iris.csv')
    iris.head()
    sns.pairplot(iris, hue='species')
    
    X  = iris.values[:, :4]
    y = iris.values[:, 4]
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)
    
    lr = LogisticRegressionCV()
    lr.fit(train_X, train_y)
    
    print("Accuracy = {:.2f}".format(lr.score(test_X, test_y)))
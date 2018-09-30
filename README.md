# Content Description
* execute `work.py` using python3.x
* folder `imdb` contains npy's
* `Report.pdf` is the work report

# Program Flow
1. import necessary modules
2. load npy's from `/imdb`
3. call `freqdist()`  
    calculate frequency distribution of id in `x_train` and `x_test`
4. call `topk()`  
    obtain frequency of top-K id in `x_train` and `x_test`
5. call `GaussianNaiveBayes()`  
    fit gnb model & calculate accuracy, precision, recall
6. repeat *step 4 ~ 5* for k = 100,1000,10000

# Program Output
``` console
Calculating freqdist of x_train & x_test...done.
~~~~~~~~~~  K = 100  ~~~~~~~~~~
Obtaining frequency of top-100 words in x_train...done.
Obtaining frequency of top-100 words in x_test...done.
Training gnb model...done.
Accuracy = 0.69168, Precision = 0.70542, Recall = 0.65824

~~~~~~~~~~  K = 1000  ~~~~~~~~~~
Obtaining frequency of top-1000 words in x_train...done.
Obtaining frequency of top-1000 words in x_test...done.
Training gnb model...done.
Accuracy = 0.81004, Precision = 0.82396, Recall = 0.78856

~~~~~~~~~~  K = 10000  ~~~~~~~~~~
Obtaining frequency of top-10000 words in x_train...done.
Obtaining frequency of top-10000 words in x_test...done.
Training gnb model...done.
Accuracy = 0.66128, Precision = 0.76809, Recall = 0.46208

Press any key to exit.
```

# `work.py`
* written in __py3__
* imports: `sklearn`, `nltk`, `numpy`
* trains a __gaussian nb model__ using __top-k most frequent words__ (k = 100,1000,1000)
* calculates __accuracy, precision, recall__ of each model


## functions in `work.py`
### `freqdist()` 
__input:__ list of id list of each sample  
__output:__ list of id freq dist for each sample
* calculates **frequency distribution of id** in each sample
* also sorts each id list in ascending order


### `topk()`
__input:__ k, list of id freq dist for each sample  
__output:__ numpy array of top-K id frequency for each sample
* obtain __frequency of top-K id__ in each sample
* i.e. number of times each word id appeared in each sample


### `GaussianNaiveBayes()`
__input:__ *none*  
__output:__ accuracy, precision, recall 
* fits a gnb model using frequency of top-K words
* calculates the accuracy, precision, recall of the model
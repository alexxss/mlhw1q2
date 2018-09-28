# Content
* execute `work.py`
* folder `imdb` contains npy's

# Program Flow
1. import necessary modules
2. load npy's from `imdb/`
3. call `freqdist()`  
    calculate frequency distribution of id in `x_train` and `x_test`
4. call `topk()`  
    obtain frequency of top-K id in `x_train` and `x_test`
5. call `GaussianNaiveBayes()`  
    fit gnb model & calculate accuracy, precision, recall
6. repeat *step 4 ~ 5* for k = 100,1000,10000

# `work.py`
* written in __py3__
* imports: `sklearn`, `nltk`,  `tqdm`, `numpy`
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
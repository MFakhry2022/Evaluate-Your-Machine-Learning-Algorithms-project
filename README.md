# Evaluate-Your-Machine-Learning-Algorithms-project
Evaluate the Performance of Machine Learning Algorithms in Python using Resampling
You need to know how well your algorithms perform on unseen data.

The best way to evaluate the performance of an algorithm would be to make predictions for new data to which you already know the answers. The second best way is to use clever techniques from statistics called resampling methods that allow you to make accurate estimates for how well your algorithm will perform on new data.
In the next, we will discover how you can estimate the accuracy of your machine-learning algorithms using resampling methods in Python and sci-kit-learn.
We must evaluate our machine learning algorithms on data that is not used to train the algorithm.
The evaluation is an estimate that we can use to talk about how well we think the algorithm may actually do in practice. It is not a guarantee of performance.
Once we estimate the performance of our algorithm, we can then re-train the final algorithm on the entire training dataset and get it ready for operational use.
Next up we are going to look at four different techniques that we can use to split up our training dataset and create useful estimates of performance for our machine-learning algorithms:
   1. Train and Test Sets.
   2. K-fold Cross Validation.
   3. Leave One Out Cross Validation.
   4. Repeated Random Test-Train Splits.
 
**1. Split into Train and Test Sets**
The simplest and very fast method we can use to evaluate the performance of a machine learning algorithm is to use different training and testing datasets.
In our example, we split the data into 67%/33% split for training and testing and evaluate the accuracy of a Logistic Regression model to be approximately 75%. 

**2. K-fold Cross Validation**
A cross-validation is an approach that you can use to estimate the performance of a machine learning algorithm with less variance than a single train-test set split.
In the example, we will be splitting the dataset into k-parts ( k=10). to get both the mean and the standard deviation of the performance measure.

**3. Leave One Out Cross Validation**
We will configure cross-validation so that the size of the fold is 1 (k is set to the number of observations in your dataset). This variation of cross-validation is called leave-one-out cross-validation.
We will see in our example results in the standard deviation score has more variance than the k-fold cross-validation results described above.

**4. Repeated Random Test-Train Splits**
Another variation on k-fold cross-validation is to create a random split of the data like the train/test split described above, but repeat the process of splitting and evaluation of the algorithm multiple times, like cross-validation.
In our example, we split the data into a 67%/33% train/test split and repeat the process 10 times to see if the distribution of the performance measure is on par with the k-fold cross-validation above.

**What Techniques to Use When**
    Generally, k-fold cross-validation is the gold standard for evaluating the performance of a machine learning algorithm on unseen data with k set to 3, 5, or 10.
    Using a train/test split is good for speed when using a slow algorithm and produces performance estimates with lower bias when using large datasets.
    Techniques like leave-one-out cross-validation and repeated random splits can be useful intermediates when trying to balance variance in the estimated performance, model training speed, and dataset size.
The best advice is to experiment and find a technique for your problem that is fast and produces reasonable estimates of performance that you can use to make decisions. If in doubt, use 10-fold cross-validation.

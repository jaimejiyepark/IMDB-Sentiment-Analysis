# IMDB-Sentiment-Analysis
## What Is It
I trained a Naive Bayes bigram classifier with IMDB reviews from [Cornell University](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
labeled positive or negative. Naives Bayes is a probablistic classifier technique that uses Bayes' Thereom  where H is the class and E is the predictor. 

![Bayes Thereom](https://i1.wp.com/www.gaussianwaves.com/gaussianwaves/wp-content/uploads/2013/10/Bayes_theorem_1.png?fit=492%2C215&ssl=1)


The classifier model was trained by computing the probability scores of each pair of words in the training data set which resulted in a binary decision between positive and negative sentiment. I also implemented the Laplace smoothing and stop-word filtering to improve the performance of the classifier. 

## How It Works
Type `python NaiveBayes.py data/imdb` into the command line to see the 10-fold Cross Validation.

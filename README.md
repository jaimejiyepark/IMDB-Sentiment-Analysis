# IMDB-Sentiment-Analysis
## What Is It
With the IMDB review data set from [Cornell University](http://www.cs.cornell.edu/people/pabo/movie-review-data/), I trained a Naive Bayes bigram classifier.
Naives Bayes is a probablistic classifier technique that uses Bayes' Thereom  where H is the class and E is the predictor. 

![Bayes Thereom](https://i1.wp.com/www.gaussianwaves.com/gaussianwaves/wp-content/uploads/2013/10/Bayes_theorem_1.png?fit=492%2C215&ssl=1)

After training the model, the program is able to analyze unclassified IMDB reviews and make a binary decision, classifying a review as either positive or negative.
In addition, I implemented the Laplace smoothing and stop-word filtering to improve the accuracy of the classifier

## How It Works
Type `python NaiveBayes.py data/imdb` into the command line to see the 10-fold Cross Validation.

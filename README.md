# Song-Analysis
This project takes in lyrics for a song and determines its genre using Machine Learning.
The training data for this algorithm are song lyrics with known genres. So we calculate
term-frequency-inverse document frequency (tf-idf) weights for every song. We calculate
the same weights for the input song. Then, we compare the input song’s weights to each 
song in the training data using cosine similarity, a measure of the similarity of two
sets of weights.

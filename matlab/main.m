%% Gabor Transformation

img = imread('iris_equal.pgm');

gaborArray = gaborFilterBank(5,8,39,39);
featureVector = gaborFeatures(img,gaborArray,4,4);
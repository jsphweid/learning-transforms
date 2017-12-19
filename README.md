# learning-transforms
My aim for this repository is to be a place to experiment with various simple transforms.

# Running
 - install tensorflow (https://www.tensorflow.org/install/)
 - `python my-tensorflow-project.py` in a terminal that has the virtualenv activated (i.e. via `source ~/tensorflow/bin/activate`)

~~simple-right-shift~~
 - ~~simple and effective NN solves this easily~~
 - ~~shifting more than 1 right (or left) seem to work just as easily for this binary example~~

drop-every-other-and-squared
 - not as straight-forward as the above problem, basic NN doesn't seem to work
 - seems to work better if they are zero padded len 6 -> len 6 instead of len 3

fourier-transform
 - first thought is that this is very possible. With a basic NN, the loss was significantly reduced
 - update: a single convolutional layer seems to do well
 - my long term goal is to -- in a larger NN that recognizes -- see what kind of transforms a NN can learn that helps it recognizes instruments or whatever... maybe the Fourier is not the best...

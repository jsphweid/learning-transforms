# learning-transforms
My aim for this repository is to be a place to experiment with various simple transforms.

# Running
 - install tensorflow (https://www.tensorflow.org/install/)
 - `python my-tensorflow-project.py` in a terminal that has the virtualenv activated (i.e. via `source ~/tensorflow/bin/activate`)

### TODO
 - various other shifts (shifting 2 or 5 places)
 - dropout, i.e. [2, 6, 1, 3, 6, 8] -> [2, 1, 6]
 - fourier transform (a buffer is the input, and the fft is the output)
 	- maybe one day a NN can find a better transform at listening that the fourier

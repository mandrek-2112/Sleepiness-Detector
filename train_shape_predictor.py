# import necessary packages
import multiprocessing
import argparse
import dlib

# parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True,
	help="path to input training XML file")
ap.add_argument("-m", "--model", required=True,
	help="path serialized dlib shape predictor model")
args = vars(ap.parse_args())

# grab the default options for dlib's shape predictor
print("[RESULT] Getting default DLIB shape predictor options.")
options = dlib.shape_predictor_training_options()

'''Here we define the tree_depth, which, as the name suggests, controls the depth of each regression tree in the 
Ensemble of Regression Trees (ERTs). There will be 2^tree_depth leaves in each tree — you must be careful to balance 
depth with speed. Smaller values of tree_depth will lead to more shallow trees that are faster, but potentially less accurate. 
Larger values of tree_depth will create deeper trees that are slower, but potentially more accurate. Typical values for 
tree_depth are in the range [2, 8].'''
options.tree_depth = 4

'''The nu option is a floating-point value (in the range [0, 1]) used as a regularization parameter to help our model generalize.
Values closer to 1 will make our model fit the training data closer, but could potentially lead to overfitting. Values closer to 
0 will help our model generalize; however, there is a caveat to the generalization power — the closer nu is to  0, the more training 
data you’ll need.'''
options.nu = 0.1

'''A series of cascades is used to refine and tune the initial predictions from the ERTs — the cascade_depth will have a dramatic 
impact on both the accuracy and the output file size of your model. The more cascades you allow for, the larger your model will 
become (but potentially more accurate). The fewer cascades you allow, the smaller your model will be (but could be less accurate).'''
options.cascade_depth = 15

'''The feature_pool_size controls the number of pixels used to generate features for the random trees in each cascade.'''
options.feature_pool_size = 400

options.num_test_splits = 50

'''Controls jitter (basic data augmentation)'''
options.oversampling_amount = 5
options.oversampling_translation_jitter = 0.1

# tells DLIB to be verbal and print training messages
options.be_verbose = True

# to use all cores of our system -- makes training faster
options.num_threads = multiprocessing.cpu_count()

# prints our training options to the terminal
print("[RESULT] Our custom options:")
print(options)

# trains
print("[RESULT] Training the model.")
dlib.train_shape_predictor(args["training"], args["model"], options)
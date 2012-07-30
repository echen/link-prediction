import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

"""
Given a training set of examples and associated features...
- Each example is of the form (src, dest, 1 if src follows dest else 0)
- Features are things like # of followers of src, Jaccard similarity
  between src and dest nodes, etc.
  
...train a machine learning classifier on this set.

Then apply this same classifier on a set of test src nodes, to form
a ranked prediction of which dest nodes each src is likely to follow.
"""

# Use this file to train the classifier.
#
# The first column in this file is the truth of a (src, dest) edge
# (i.e., 1 if the edge is known to exist, 0 otherwise).
# The rest of the columns are features on that edge.
TRAINING_SET_WITH_FEATURES_FILENAME = "my_data/my_ml_training_set_with_features.csv"

# This file contains candidate edge pairs to score, along with
# features on these candidate edges.
#
# The first column is the src node, the second is the dest node,
# the rest of the columns are features.
CANDIDATES_TO_SCORE_FILENAME = "my_data/my_candidates_with_features.csv"

########################################
# STEP 1: Read in the training examples.
########################################
truths = [] # A truth is 1 (for a known true edge) or 0 (for a false edge).
training_examples = [] # Each training example is an array of features.
for line in open(TRAINING_SET_WITH_FEATURES_FILENAME):
  fields = [float(x) for x in line.split(",")]
  truth = fields[0]
  training_example_features = fields[1:]

  truths.append(truth)  
  training_examples.append(training_example_features)

#############################
# STEP 2: Train a classifier.
#############################
clf = RandomForestClassifier(n_estimators = 500, compute_importances = True, oob_score = True)
clf = clf.fit(training_examples, truths)

###############################
# STEP 3: Score the candidates.
###############################
BATCH_SIZE = 10000
src_dest_nodes = []
examples = []
predictions = []
for line in open(CANDIDATES_TO_SCORE_FILENAME):
  fields = [float(feature) for feature in line.split(",")]
  
  src = fields[0]
  dest = fields[1]
  src_dest_nodes.append((src, dest))
  
  example_features = fields[2:]
  examples.append(example_features)
  
  if len(examples) == BATCH_SIZE:
    predictions = clf.predict_proba(examples)
    for i in xrange(batch_size):
      print ",".join([str(x) for x in [src_dest_nodes[i][0], src_dest_nodes[i][1], predictions[i][1]]])
    examples = []
    predictions = []
    src_dest_nodes = []
    
predictions = clf.predict_proba(examples)  
for i in xrange(len(predictions)):
  print ",".join([str(x) for x in [src_dest_nodes[i][0], src_dest_nodes[i][1], predictions[i][1]]])
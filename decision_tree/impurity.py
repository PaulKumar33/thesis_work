import numpy as np
import pandas as pd

df = pd.read_excel("../decision_data.xlsx", "all_features-double_node")
#df = df.drop(columns=['figure'])

df_mat = df.values.tolist()

def class_counts(df):
    '''
    this method gets the number of labels in a passed in data frame
    '''

    counts = {}
    for row in df:
        label = row[-1]
        if(label not in counts.keys()):
            counts[label] = 0
        counts[label] += 1

    return counts

########
#Test counts
#print(class_counts(excel))
########

#just a reminder. when at a node, lets say root, we go thrugh all the labels and determine which
#feature produces the lowest gini impurity. this the becomes a splotting feature that splits the data
#well

def unique_vals(rows, col):
    """finds the unique values in each col"""
    return set([row[col] for row in rows])


######
#test uniqure
#print(unique_vals(excel.values.tolist(), 0))

def is_numeric(value):
    """checks to see if the value is numeric"""
    return isinstance(value, int) or isinstance(value, float)

def question(row, numerical, limit=None):
    if(numerical):
        return row > limit
    else:
        return row == limit


class Question:
    def __init__(self,col, val):
        self.col = col
        self.val = val

    def match(self, example):
        """this method compares the feature value in a question to the feature value in the example"""
        val = example[self.col]
        if(is_numeric(val)):
            return val>=self.val
        else:
            return val == self.val

    def __repr__(self):
        """this method returns a string of the object created"""

        condition = "=="
        if(is_numeric(self.val)):
            condition = ">="
        return f"Is {df.columns[self.col]} {condition} {self.val}"



"""q = Question(1, 1)
example = df_mat[0]
print(q.match(example))"""

def partition(rows, question):
    """
    partion the data set depending on the result of the comparison

    recall, each data point gets tested at a node when building the tree. We split this result
    depending on the result of the question. the gini is then calculated based on the partition
    """
    true_rows, false_rows = [], []
    for row in rows:
        if(question.match(row)):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


###############
#test the partition
#note for future: if we one hot encode, make sure the limit for numeric values is 1
"""q = Question(0, 1)
print(parition(df_mat, q))"""

def gini(rows):
    """calculate the gini impurity for each row"""

    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob = counts[label]/len(rows)
        impurity -= prob**2

    return impurity

#####################################
#temp_df = [['left'], ['left'], ['left'], ['left'], ['right'],['left'], ['left'], ['left'], ['left'], ['left'], ['right'], ['left']]
#temp_df = [[item[-1]] for item in df_mat]
#print(temp_df)
#print(gini(temp_df))
#####################################

def infor_gain(left, right, current_uncertainty):
    """calculate the uncertainty and then determine the weighted uncertainity of the two nodes"""

    p = float(len(left))/(float(len(left))+float(len(right)))
    return current_uncertainty - (p*gini(left) + (1-p)*gini(right))

def find_best_split(rows):
    """this findds the best question for the node. it iterates over all the features and figures out which results in the
    best impurity and information gains from the data"""
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)

            if(len(true_rows) == 0 or len(false_rows) == 0):
                continue

            #gain of information
            gain = infor_gain(true_rows, false_rows, current_uncertainty)

            if(gain>best_gain):
                best_gain, best_question = gain, question

    return best_gain, best_question


##################################
#print(df_mat)
#gain, _question = find_best_split(df_mat)
#print(gain, _question)


class Leaf:
    """A Leaf node classifies data.
    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question.
    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """Builds the tree.
    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    gain, question = find_best_split(rows)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    true_rows, false_rows = partition(rows, question)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def return_model(model):
    print_tree(model)
    return model

tree = build_tree(df_mat)
print_tree(tree)
#_classify = classify([2.192382813,	2.8125,	0.099384282,	2.490029405,	1.213396868,	'no-cross-left'], tree)

'''max_guess = 0
max_class = None
for _class_ in _classify:
    if(_classify[_class_]>max_guess):
        max_class, max_guess = _class_, _classify[_class_]

print(f"Predicted: {max_class}")'''



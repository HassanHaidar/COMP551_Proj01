import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import operator
import matplotlib.pyplot as plt
import time

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

# Example:
data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
# for info_name, info_value in data_point.items():
#    print(info_name + " : " + str(info_value))

# Split the data as training, validation, and test sets
training_set = data[:10000] 

validation_set = data[10000:11000] 

test_set = data[11000:12000]


#  returns lowercase string 
def preprocess_text(string):
    string = string.lower()
    return string.split()
    

# returns a list of the 160 most occuring 
def most_occuring_words(data, k):
    word_list = {}
    for instance in data:
        text_list = instance["text"].lower().split() # turn strings to lower case
        for word in text_list:
            if not (word in word_list):
                word_list[word] = 1
            else: 
                word_list[word] += 1;

    # sort words by increasing order
    sorted_words = sorted(word_list.items(), key = lambda kv : kv[1])

    # take the 160 most occuring words
    sorted_words = sorted_words[len(sorted_words): len(sorted_words) - k - 1:-1]
    return sorted_words



sorted_words = most_occuring_words(data, 160)



# given a string list, return the 160-feature vector as a numpy array
def count_words(comment):
    # comment = comment.split()
    # feature = np.zeros(160)
    feature = [0.0] * len(sorted_words);
    for i in range(len(sorted_words)):
        for word in comment:
            if sorted_words[i][0] == word:
                feature[i] = feature[i] + 1
    return feature



# extract the children, controversiality,  is_root
def extract_features(dictionary):
    list = []
    # get controversialit feature
    list.append(float(dictionary['controversiality']))
    # if comment is root, add 1.0
    # else append 0.0
    if dictionary['is_root']:
        list.append(1.0)
    else:
        list.append(0.0)
    # extract number of replies in teh thread to this comment
    list.append(float(dictionary['children']))
    return list


# method that returns the number of punctuation in a string
def count_punc(string):
    punc = ['!','?', '.', ',']
    count = 0
    for i in range(len(string)):
        if string[i] in punc:
            count += 1
    return count



# create target/output vector for the data set given as dictionary
def get_target(data):
    target = [instance['popularity_score'] for instance in data]
    target = np.array(target)
    return target

Y_train = get_target(training_set) 
Y_val = get_target(validation_set)
Y_test = get_target(test_set)


# create matrix out of a dictionary of data
# input: list of dictionaries
# output: matrix X (input), vector y (target)
def preprocess(data):
    matrix = [];
    for i in range(len(data)):
        # get dictionary
        instance = data[i]

        
        

        # get recurrence feature
        # recurrence_feature = count_words(preprocess_text(instance['text']))
        # get other features
        features = extract_features(instance)
        # features.extend(recurrence_feature)

        # add comment length as a feature
        # features.append(float(len(instance['text'])))

        # add number of punctuation marks as comment
        features.append(float(count_punc(instance['text'])))

        features.append(1.0)
        matrix.append(features)
        
    return np.array(matrix)



# perform linear regression on X
# X is input, Y is target
def linear_regression(X, Y):
    start_regression = time.time()
    Xt = X.T
    Xtx = Xt.dot(X)
    inverse_Xtx = np.linalg.inv(Xtx)
    W = np.dot(np.dot(inverse_Xtx, Xt), Y)
    
    return W


# predict popularity given new data
# input: set of weights
# returns prediction and MSE of prediction
def predict(X, W, Y):
    predicted_Y = X.dot(W);
    mse = (np.square(predicted_Y - Y)).mean()
    return predicted_Y, mse


#==================================PROGRAM===========================

X = preprocess(training_set)

start_regression = time.time()
W = linear_regression(X,Y_train)
end_regression = time.time()

time_regression = end_regression - start_regression

mse_training = predict(X, W, Y_train)[1]

X_val = preprocess(validation_set)
mse_validation = predict(X_val, W, Y_val)[1]

X_test = preprocess(test_set)
mse_testing = predict(X_test, W, Y_test)[1]


print("Training Error: ", mse_training)
print("Validation Error: ", mse_validation)
print("Test Error: ", mse_testing)



mse_descent_train = []
mse_descent_val = []
def gradient_descent(X,X_val, Y, Y_val, W, betta = 0.0, etta = 0.00001, EPSILON = 0.0000005):
    Xt = X.T
    Xtx = Xt.dot(X)
    Xty = Xt.dot(Y)
    k = 0
    f.write("STARTING LEARNING RATE: {} --- NUMBER OF FEATURES IS {} --- STARTING WEIGHT IS {} \n".format(etta, Xt.shape[0], W[0]))
    while True:
        W_prev = W
        lr_rate = etta / (1 + betta)
        W = W_prev - 2 * lr_rate * (Xtx.dot(W) - Xty)
        betta += 1

        # training error
        mse = predict(X, W, Y)[1]
        mse_descent_train.append(mse)

        # validation error
        mse_val = predict(X_val, W, Y_val)[1]
        mse_descent_val.append(mse_val)

        log = "iteration {} :  training error = {} | validation error = {} \n".format(k, mse, mse_val)
        print(log)
        f.write(log)
        k = k + 1
        condition =  pow(np.linalg.norm(W - W_prev, 2), 2)
        
        if (condition < EPSILON) or (k > 1000):
            break

f = open("report.txt", "w")
f.write("CLOSED FORM SOLUTION: \n")
f.write("Training Error: {} \n".format(mse_training))
f.write("Validation Error: {} \n".format(mse_validation))
f.write("Test Error: {} \n".format(mse_testing))

f.write("GRADIENT DESCENT: \n")

start_descent = time.time()
gradient_descent(X, X_val, Y_train, Y_val, np.zeros(X.shape[1]))
end_descent = time.time()

plt.title("MSE loss per iteration")
plt.plot(mse_descent_train, 'b')
plt.plot(mse_descent_val, 'r')
plt.ylabel('MSE')
plt.xlabel('iteration')
plt.legend(['training error', 'validation error'])
plt.show()

time_descent = end_descent - start_descent

f.write("closed from runtime: {} seconds | gradient descent runtime: {} seconds".format(time_regression,time_descent))
f.close()
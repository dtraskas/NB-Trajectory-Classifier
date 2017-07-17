import csv
import random
import math

def load_data(filename, bIsX):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        dataset = list(reader)

    if bIsX:
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]] 
    else:
        for i in range(len(dataset)):
            dataset[i] = [x for x in dataset[i]] 

    return dataset

def split_data(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    
    return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

train_x = load_data('/Users/talos/Projects/data/nd013_pred_data/train_states.txt', True)
train_y = load_data('/Users/talos/Projects/data/nd013_pred_data/train_labels.txt', False)
test_x = load_data('/Users/talos/Projects/data/nd013_pred_data/test_states.txt', True)
test_y = load_data('/Users/talos/Projects/data/nd013_pred_data/test_labels.txt', False)

dataset = [[1,20,1], [2,21,0], [3,22,1]]
separated = separate_by_class(dataset)
print(separated)
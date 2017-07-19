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
            dataset[i] = [x.strip("[").strip("]").strip("'").strip("\"") for x in dataset[i]] 

    return dataset

def split_data(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def separate_by_class(dataset, class_y):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        key = str(class_y[i])
        if (key not in separated):
            separated[key] = []
        separated[key].append(vector)
    
    return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarise(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    return summaries

def summarise_by_class(dataset, class_y):
    separated = separate_by_class(dataset, class_y)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarise(instances)
    
    return summaries

def calculate_prob(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculate_prob_by_class(summaries, input_vector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = input_vector[i]
            probabilities[classValue] *= calculate_prob(x, mean, stdev)

    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_prob_by_class(summaries, input_vector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def get_predictions(summaries, test):
    predictions = []
    for i in range(len(test)):
        result = predict(summaries, test[i])
        predictions.append(result)
    
    return predictions

def get_accuracy(test, predictions):
    correct = 0
    for x in range(len(test)):
        testy = test[x][0]
        predy = predictions[x].strip("[").strip("]").strip("'")

        if testy == predy:
            correct += 1
    
    return (correct/float(len(test))) * 100.0

train_x = load_data('data\\train_states.txt', True)
train_y = load_data('data\\train_labels.txt', False)
test_x = load_data('data\\test_states.txt', True)
test_y = load_data('data\\test_labels.txt', False)

summaries = summarise_by_class(train_x, train_y)
# test model
predictions = get_predictions(summaries, test_x)
accuracy = get_accuracy(test_y, predictions)

print('Accuracy: ' + str(accuracy))

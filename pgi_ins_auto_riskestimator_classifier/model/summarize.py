import math
from sklearn import preprocessing


def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stddev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers))
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stddev(attribute), min(attribute), max(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    print('summarize_by_class :' + str(summaries))
    return summaries

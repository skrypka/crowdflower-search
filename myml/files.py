import os
import csv
import random
import cPickle as pickle

def load(path):
    return pickle.load(open(path))

def dump(path, data):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)

    with open(path, "w") as f:
        pickle.dump(data, f)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def subsample(input_file, output_file, P = 0.5):
    i = open( input_file )
    o = open( output_file, 'w' )

    reader = csv.reader(i)
    writer = csv.writer(o)

    headers = reader.next()
    writer.writerow(headers)

    for line in reader:
        r = random.random()
        if r > P:
            continue
        writer.writerow(line)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    mnist = input_data.read_data_sets('data/', one_hot=True)

if __name__ == '__main__':
    main()
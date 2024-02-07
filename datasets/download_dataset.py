import wget
import tarfile

wget.download("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",out="./data/")

file = tarfile.open('./data/cifar-10-python.tar.gz')
file.extractall('./data/cifar-10')
file.close()

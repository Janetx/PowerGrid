import argparse, sys, os, logging
import pandas as pd
import powergrid_data

parser = argparse.ArgumentParser(description='Anomaly Detection')
parser.add_argument("-ts", "--test", dest="test", default=os.path.join("data", "test", "test_v1.csv"), help="File containing testing data (default=data/test)")
parser.add_argument("-tr", "--train", dest="train", default=os.path.join("data", "train", "train.csv"), help="File containing training data (default=data/train)")
parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
args = parser.parse_args()


def timeit(mainFunction):
    import time
    from functools import wraps

    @wraps(mainFunction)
    def function_wrapper(*args, **kwargs):
        start = time.clock()
        mainFunction(*args, **kwargs)
        end = time.clock()
        logging.debug("\nElapsed Time: %s \n" % (end - start,))

    return function_wrapper


@timeit
def main():
    power_grid = powergrid_data.datasets(args.train, args.test)
    # print power_grid.data #training data
    # print power_grid.target #testing data

if __name__ == '__main__':
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    main()


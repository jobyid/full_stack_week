import argparse
from scripts import model_funcs as mf



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Some sort of CNN working it's magic, using pretrained resnet 18", epilog="That is how it is done")
    parser.add_argument("--test", help="some sort of test fucntion for running this stuff")
    parser.add_argument("--download_data",nargs='?', default='d', help="download the data to train on, provide a path to downlaod from or use our default set")
    parser.add_argument("--unzip",nargs='?', default='d', help="unzip the data set")
    parser.add_argument("--train")
    parser.add_argument("--epochs")
    parser.add_argument("--optimiser")
    args = parser.parse_args()
    print(args)
    if args.test == 'hh':
        mf.test_func(args.test)
    if args.download_data is None:
        mf.download_data()
    elif args.download_data is not None:
        mf.download_data(args.download_data)
    if args.unzip is not None:
        mf.unzip()
    # --train --epochs 2
    # --optimizer str()
    # --load-weights Path()
    # --test

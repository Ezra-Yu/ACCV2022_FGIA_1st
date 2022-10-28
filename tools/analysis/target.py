import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument('pkl', help='Ensemble results')
    args = parser.parse_args()
    return args

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def main():
    args = parse_args()
    data = load_pkl(args.pkl)
    print(f"{len(data)} samples have been found....")

if __name__ == '__main__':
    main()
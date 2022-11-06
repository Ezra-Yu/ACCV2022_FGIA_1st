import numpy as np
import pickle
import argparse
from collections import defaultdict

CLASSES = [f"{i:0>4d}" for i in range(5000)]

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument('pkl', help='Ensemble results')
    parser.add_argument('--thr', default=None, type=float, help='threshold')
    parser.add_argument('--out', default="pseudo.txt", help='output path')
    args = parser.parse_args()
    return args

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def plot_labels(counts, numbers):
    import matplotlib.pyplot as plt

    plt.bar(counts, numbers)
    plt.savefig("target_label.jpg")
    plt.show()

def plot_scores(scores):
    import matplotlib.pyplot as plt

    med = np.median(scores)
    scores_ = np.sort(scores)
    assert scores is not scores_
    plt.ylim((0, 1))
    plt.plot(range(len(scores_)), scores_)
    plt.hlines(med, 0, len(scores_), colors="red")
    plt.text(0, med, f'median : {med}')
    plt.savefig("scores.jpg")
    plt.show()

def generate_pseudo_label(data, scores, thr):
    pseudo_list = []
    for (filename, classname, _), pred_score in zip(data, scores):
        if pred_score > thr:
            pseudo_list.append( (filename, classname) )
    return pseudo_list


def main():
    args = parse_args()
    data = load_pkl(args.pkl)
    print(f"{len(data)} samples have been found....")

    data_dict = defaultdict(list)
    for i, (filename, classname, score) in enumerate( data ):
        data_dict[classname].append(i)

    max_counts = 0
    for classname in CLASSES:
        max_counts = max(max_counts, len(data_dict[classname]))

    counts = list(range(max_counts + 1))
    count_dict = defaultdict(int)
    for i in counts:
        count_dict[i] = 0
    for classname in CLASSES:
        count_dict[len(data_dict[classname])] += 1

    numbers = list(count_dict.values())
    plot_labels(counts, numbers)

    scores = np.array([np.max(score) for i, (filename, classname, score) in enumerate( data )])
    plot_scores(scores)

    # thr = np.median(scores)
    if args.thr is None:
        args.thr = np.median(scores)
    pseudo_list = generate_pseudo_label(data, scores, args.thr)
    print(f"Get {len(pseudo_list)} pseudo samples....")

    assert args.out and args.out.endswith(".txt")
    with open(args.out, "w") as outfile:
        for filename, label in pseudo_list:
            outfile.write(f"testb/{filename} {label}\n")

if __name__ == '__main__':
    main()

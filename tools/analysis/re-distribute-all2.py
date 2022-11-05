import numpy as np
import csv
import pickle
import argparse
from collections import defaultdict


CLASSES = [f"{i:0>4d}" for i in range(5000)]

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble Learning')
    parser.add_argument('--pkla', help='Ensemble results')
    parser.add_argument('--pklb', help='Ensemble results')
    parser.add_argument('--L', default=27, type=int, help='Ensemble results')
    parser.add_argument('--H', default=33, type=int, help='Ensemble results')
    parser.add_argument('--out', default="pred_results.csv", help='output path')
    args = parser.parse_args()
    return args

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def convert_data_dict2list(data_dict):
    result_list = []
    for filename, scores in data_dict.items():
        pred_label = np.argmax(scores)
        pred_class = CLASSES[pred_label]
        result_list.append( [filename, pred_class, scores] )
    return result_list


def extract_data(data_list, L, H):
    """获取以下有用的数据形式

    class2sampleIdx_dict: {classes: sampleidx_list}
    less_count_classes : 被预测次数小于 L 的类别list
    less_count_classes : 被预测次数大于 H 的类别list
    
    """
    print(f"{len(data_list)} samples have been found....")

    class2sampleIdx_dict = defaultdict(list)
    for i, (filename, classname, score) in enumerate( data_list ):
        class2sampleIdx_dict[classname].append(i)

    max_counts = 0
    for classname in CLASSES:
        max_counts = max(max_counts, len(class2sampleIdx_dict[classname]))

    less_count_classes = defaultdict(list)
    large_count_classes = defaultdict(list)
    for classname in CLASSES:
        if len(class2sampleIdx_dict[classname]) < L:
            less_count_classes[len(class2sampleIdx_dict[classname])].append(classname)
        elif len(class2sampleIdx_dict[classname]) > H:
            large_count_classes[len(class2sampleIdx_dict[classname])].append(classname)

    return  class2sampleIdx_dict, less_count_classes, large_count_classes


def plot_scores(scores):
    """画出最大 score 的分布, 返回中位数"""
    import matplotlib.pyplot as plt

    med = np.median(scores)  
    scores_ = np.sort(scores)
    assert scores is not scores_
    # plt.ylim((0, 1))
    plt.plot(range(len(scores_)), scores_)
    plt.hlines(med, 0, len(scores_), colors="red")
    plt.text(0, med, f'median : {med}')
    plt.savefig("scores.jpg")
    plt.show()
    return med


def plot_labels(class2sampleIdx_dict, dscp):
    """画出label的分布"""
    max_counts = 0
    for classname in CLASSES:
        max_counts = max(max_counts, len(class2sampleIdx_dict[classname]))
    
    counts = list(range(max_counts + 1))
    count_dict = defaultdict(int)
    for i in counts:
        count_dict[i] = 0
    for classname in CLASSES:
        count_dict[len(class2sampleIdx_dict[classname])] += 1

    numbers = list(count_dict.values())
    import matplotlib.pyplot as plt

    plt.bar(counts, numbers)
    plt.savefig(f"{dscp}.jpg")
    plt.show()


def main():
    args = parse_args()
    L = args.L
    H = args.H
    
    data_dicta = load_pkl(args.pkla) if args.pkla else dict()
    data_dictb = load_pkl(args.pklb) if args.pklb else dict()
    data_dict = {**data_dicta, **data_dictb}

    result_list = convert_data_dict2list(data_dict) # result_list is list of [filename, pred_class, scores]
    (class2sampleIdx_dict, less_count_classes, 
        large_count_classes) = extract_data(result_list, L, H)
    plot_labels(class2sampleIdx_dict, "target_label_before")
    
    all_scores= np.stack([r[2] for r in result_list], axis=0)  # (N, M) N = num_samplers, M = num_classes
    all_pred_labels = np.array([int(r[1]) for r in result_list])  # 所有sample的 pred_labels
    all_pred_scores = all_scores.max(axis=1)                      # 所有sample的 pred_scores
    assert len(all_pred_labels) == len(all_pred_scores), f"{len(all_pred_labels)} != {len(all_pred_scores)}"
    med = plot_scores(all_pred_scores)
    
    print("Process less count classes")
    for count, classname_list in less_count_classes.items():
        for classname in classname_list:
            class_idx = int(classname)
            scores = all_scores[:, class_idx]  # 预测该类别的所有分数
            scores[all_pred_labels==class_idx] = 0  # 排除掉预测为该classname的sample
            topk = L - count                                  # 需补偿的个数 = L - cur_count
            indxs = np.argpartition(scores, -topk)[-topk:]
            for ind in indxs:
                if all_pred_scores[int(ind)] > med:  # 当其本身的预测分数非常高的时候，不做处理
                    continue
                else:
                    result_list[int(ind)][1] = classname

    print("Process large count classes")
    for count, classname_list in large_count_classes.items():
        for classname in classname_list:
            class_idx = int(classname)
            scores = all_scores[:, class_idx]
            sample_idxs = class2sampleIdx_dict[classname]
            # 预测为当前 classname 的所有samper的得分， 取负方便取最小的topk
            samper_scores = -1 * scores[sample_idxs]
            topk = count - H                           # 需删除的个数 = cur_count - H
            indxs = np.argpartition(samper_scores, -topk)[-topk:]  # 因为取负了，这里还是使用topk得到预测分数最小的topk个
            for ind in indxs:
                sample_idx = sample_idxs[ind]
                scores = all_scores[sample_idx,:]
                if scores[class_idx] > med:               # 如果预测当前classname时，分数比较高，则不做处理
                    continue
                scores[class_idx] = 0                    # 否则，取剩下预测的最高预测
                new_pred_label = np.argmax(scores)
                result_list[sample_idx][1] = CLASSES[new_pred_label]

    class2sampleIdx_dict_after = defaultdict(list)
    for i, (filename, classname, score) in enumerate( result_list ):
        class2sampleIdx_dict_after[classname].append(i)

    plot_labels(class2sampleIdx_dict_after, "target_label_after")

    assert args.out and args.out.endswith(".csv") 
    with open(args.out, "w") as csvfile:
        writer = csv.writer(csvfile)
        for result in result_list:
            if result[0] in data_dictb:
                writer.writerow(result[:2])


main()
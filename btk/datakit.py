"""
Tools for working with data and machine learning models
"""

import os
import numpy as np
import pickle
import random
import time

def pickle_set(trainx: np.ndarray, trainy: np.ndarray, valx: np.ndarray, valy: np.ndarray, datadir: str) -> str:
    """
    Create pickle files for a machine learning data set

    Args:
        trainx (np.ndarray): Training data
        trainy (np.ndarray): Training labels
        valx (np.ndarray): Validation data
        valy (np.ndarray): Validation labels
        dir (str): Target directory

    Returns:
        str: Identifier for locating files
    """
    timecode = round(time.time())
    with open(f"{os.environ['dstore']}{datadir}\\trainx-{timecode}", 'ab') as pkf:
        pickle.dump(trainx, pkf)
    with open(f"{os.environ['dstore']}{datadir}\\trainy-{timecode}", 'ab') as pkf:
        pickle.dump(trainy, pkf)
    with open(f"{os.environ['dstore']}{datadir}\\valx-{timecode}", 'ab') as pkf:
        pickle.dump(valx, pkf)
    with open(f"{os.environ['dstore']}{datadir}\\valy-{timecode}", 'ab') as pkf:
        pickle.dump(valy, pkf)
    return timecode

def depickler(trainx: str, trainy: str, valx: str, valy: str, datadir: str) -> tuple[str, str, str, str]:
    """
    Loads a set of pickle files for training neural networks

    Args:
        trainx (str): Training data file path
        trainy (str): Training labels file path
        valx (str): Validation data file path
        valy (str): Validation labels file path
        dir (str): Target directory

    Returns:
        tuple[str, str, str, str]: Data sets
    """
    with open(f"{os.environ['dstore']}{datadir}\\{trainx}", 'rb') as pkf:
        trainxx = pickle.load(pkf)
    with open(f"{os.environ['dstore']}{datadir}\\{trainy}", 'rb') as pkf:
        trainyy = pickle.load(pkf)
    with open(f"{os.environ['dstore']}{datadir}\\{valx}", 'rb') as pkf:
        valxx = pickle.load(pkf)
    with open(f"{os.environ['dstore']}{datadir}\\{valy}", 'rb') as pkf:
        valyy = pickle.load(pkf)
    return trainxx, trainyy, valxx, valyy

def mlstats(nhist, measure, tm1=None, ohist=None, tm2=None):
    """Show stats from ML training runs, one or two cycles averaged"""
    score = (max(nhist.get(f"val_{measure}")) + (2.7182818**-(min(nhist.get("val_loss")))) + (sum(nhist.get(f"val_{measure}")[-7:]) / 7) + (sum(2.7182818**-(np.array(nhist.get("val_loss")[-7:]))) / 7)) / 4
    bvacc = max(nhist.get(f"val_{measure}"))
    bvloss = min(nhist.get("val_loss"))
    avacc = sum(nhist.get(f"val_{measure}")[-7:]) / 7
    avloss = sum(nhist.get("val_loss")[-7:]) / 7
    if tm1:
        samps = len(nhist.get(f"val_{measure}")) * 32 * (192 + 64) / (tm1)
    if ohist is None:
        print(f'Score: {round(score, 6)}')
        print(f'Best VAccuracy: {round(bvacc, 6)}')
        print(f'Best VLoss: {round(bvloss, 6)}')
        print(f'Last 7 Avg VAccuracy: {round(avacc, 6)}')
        print(f'Last 7 Avg VLoss: {round(avloss, 6)}')
        if tm1:
            print(f'Training time: {round(tm1, 6)}')
            print(f'Samples per second: {round(samps, 6)}')
    else:
        oscore = (max(ohist.get(f"val_{measure}")) + (2.7182818**-(min(ohist.get("val_loss")))) + (sum(ohist.get(f"val_{measure}")[-7:]) / 7) + (sum(2.7182818**-(np.array(ohist.get("val_loss")[-7:]))) / 7)) / 4
        obvacc = max(ohist.get(f"val_{measure}"))
        obvloss = min(ohist.get("val_loss"))
        oavacc = sum(ohist.get(f"val_{measure}")[-7:]) / 7
        oavloss = sum(ohist.get("val_loss")[-7:]) / 7
        if tm1 and tm2:
            osamps = len(ohist.get(f"val_{measure}")) * 32 * (192 + 64) / (tm2)
        print(f'Score: {round((score + oscore) / 2, 6)}')
        print(f'Best VAccuracy: {round((obvacc + bvacc) / 2, 6)}')
        print(f'Best VLoss: {round((obvloss + bvloss) / 2, 6)}')
        print(f'Last 7 Avg VAccuracy: {round((oavacc + avacc) / 2, 6)}')
        print(f'Last 7 Avg VLoss: {round((oavloss + avloss) / 2, 6)}')
        if tm1 and tm2:
            print(f'Training time: {round((tm1 + tm2) / 2, 6)}')
            print(f'Samples per second: {round((osamps + samps) / 2, 6)}')

def model_stats(predictions):
    print('\nLabel\t\tRecall\t\tPrecision\tF1\t\tRatio')
    metrics = []
    total = len(predictions)
    for x in range(19):
        tp = 0
        guesses = [y for y in predictions if y[1] == x]
        actual = [y for y in predictions if y[0] == x]
        for z in actual:
            if z[0] == z[1]:
                tp += 1
        alen = len(actual)
        glen = len(guesses)
        if glen == 0 or alen == 0 or tp == 0:
            alen += 1
            glen += 1
            tp += 1
        recall = round(tp / alen, 3)
        precis = round(tp / glen, 3)
        f1 = round(2 * (precis * recall) / (precis + recall), 3)
        metrics.append((alen, glen, tp))
        print(f'{x}\t\t{recall}\t\t{precis}\t\t{f1}\t\t{tp}/{glen}/{alen}')
    correct = sum([x[2] for x in metrics])
    mprec = round(np.mean([x[2] / x[1] for x in metrics]), 4)
    mrec = round(np.mean([x[2] / x[0] for x in metrics]), 4)
    f1mac = round(2 * ((mprec * mrec) / (mprec**-1 + mrec**-1)), 4)
    f1mic = round(correct / total, 4)
    mcc = round((correct * total - sum([x[0] * x[1] for x in metrics])) / np.sqrt((total**2 - sum(x[0]**2 for x in metrics))*(total**2 - sum(x[1]**2 for x in metrics))), 4)
    #ck = round((correct * total - sum([x[1] * x[0] for x in metrics])) / (total**2 - sum([x[1] * x[0] for x in metrics])), 3)
    print(f"\nMacro Precision\t\t{mprec}\nMacro Recall\t\t{mrec}\nF1 Macro\t\t{f1mac}\nF1 Micro\t\t{f1mic}\nMCC\t\t\t{mcc}")
    #print(f'CK\t\t\t{ck}')
    print(correct, '/', total)
    return mprec, mrec, f1mac, f1mic, mcc

def zshuffle(data, labels):
    """shuffle seperate data sets while maintaining cohesion"""
    temp = list(zip(data, labels))
    random.shuffle(temp)
    data = [x[0] for x in temp]
    labels = [x[1] for x in temp]
    return np.array(data), np.array(labels)

class DataGen:
    """
    Custom data generator class for pytorch
    """
    def __init__(self, data, labels, batch_len=16):
        self.labels = labels
        self.data = data
        self.batch_len = batch_len
        self.i = 0
        self.maxi = round(len(self.data) / self.batch_len) - 2

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.i >= self.maxi:
            raise StopIteration
        self.i += 1
        return self.data[(self.i - 1) * self.batch_len:self.i * self.batch_len], self.labels[(self.i - 1) * self.batch_len:self.i * self.batch_len]

    """ def __getitem__(self, idx):
        return torch.from_numpy(self.data[[idx]]), torch.as_tensor(self.labels[[idx]], dtype=torch.float) """
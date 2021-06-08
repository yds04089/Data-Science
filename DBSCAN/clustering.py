import pandas as pd
import numpy as np
import time
import sys

UNASSIGNED = 0
NOISE = -1


# calculate distance between two points
def distance(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**(1/2)


# make a list of neighbor points
def get_neighbor(data, X, Eps):
    neighbor = []
    for i in data:
        if (distance(i[1], i[2], X[1], X[2]) <= Eps):
            neighbor.append(i)
    return neighbor


# cluster의 크기가 큰 순으로 order를 만듬
def make_order(label):
    cnt = {}
    # cluster의 크기를 알기 위해 cnt에 cluster당 point개수를 저장함
    for val in label:
        if val not in cnt:
            cnt[val] = 1
        else:
            cnt[val] += 1
    order = sorted(cnt.items(), reverse=True, key=lambda item: item[1])
    return order


# n개만큼의 output file을 write함
def write_file(label, inputname, n, order):
    cnt = 0
    # 크기가 더 큰 cluster의 순으로 file을 만들어 write함
    for key, val in order:
        # Noise이면 무시함
        if key == NOISE:
            continue
        # n개만큼의 file을 write하면 종료함
        if n == cnt:
            break
        id = key
        cluster = []
        for idx in range(len(label)):
            if label[idx] == id:
                cluster.append(idx)
        file_name = f'./test-3/{inputname}_cluster_{cnt}.txt'
        f = open(file_name, 'w')
        for value in cluster:
            f.write(str(value)+'\n')
        f.close()
        cnt += 1
    # if number of clusters < n, then make more empty files
    while cnt < n:
        file_name = f'./test-3/{inputname}_cluster_{cnt}.txt'
        f = open(file_name, 'w')
        f.close()
        cnt += 1


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print(
            "USAGE: python clustering.py <input filename> <number of clusters> <Eps> <MinPts>")
        sys.exit()
    filename = f'./data-3/{sys.argv[1]}'
    n = int(sys.argv[2])
    Eps = int(sys.argv[3])
    MinPts = int(sys.argv[4])
    cluster_id = 0
    df = pd.read_csv(filename, sep='\t', header=None)
    data = df.values.tolist()
    label = [UNASSIGNED]*len(data)
    start_time = time.time()
    # scan all data
    for X in data:
        now_id = int(X[0])
        # unassigned point가 아니라면 continue
        if label[now_id] != UNASSIGNED:
            continue
        neighbor = get_neighbor(data, X, Eps)
        # if core point, then make new cluster
        if len(neighbor) >= MinPts:
            now_cluster = []
            now_cluster.extend(neighbor)
            cluster_id += 1
            # 첫번째 neighbor points의 label들을 cluster_id로 assign함
            for nc in now_cluster:
                if label[int(nc[0])] == UNASSIGNED:
                    label[int(nc[0])] = cluster_id
            # 더이상 확장할 수 없을 때까지 cluster를 계속 확장함
            for i in now_cluster:
                next_id = int(i[0])
                # if noise point
                if label[next_id] == NOISE:
                    label[next_id] = cluster_id
                # if unassigned point
                elif label[next_id] == UNASSIGNED:
                    label[next_id] = cluster_id
                    new_neighbor = get_neighbor(data, i, Eps)
                    # if core point
                    if len(new_neighbor) >= MinPts:
                        for nn in new_neighbor:
                            if label[int(nn[0])] != cluster_id:
                                now_cluster.append(nn)
                # if point is already assigned in another cluster
                elif label[next_id] != cluster_id:
                    # if core point
                    if len(get_neighbor(data, i, Eps)) >= MinPts:
                        diff_cluster_id = label[next_id]
                        for t in range(len(label)):
                            if label[t] == diff_cluster_id:
                                label[t] = cluster_id
                                now_cluster.append(data[t])
                # if point is already assigned in now_cluster
                elif label[next_id] == cluster_id:
                    new_neighbor = get_neighbor(data, i, Eps)
                    # if core point
                    if len(new_neighbor) >= MinPts:
                        for nn in new_neighbor:
                            if label[int(nn[0])] == cluster_id:
                                continue
                            else:
                                label[int(nn[0])] = cluster_id
                                now_cluster.append(nn)
        # if not core point, then NOISE
        else:
            label[now_id] = NOISE
    print(len(set(label)))
    print(time.time()-start_time)
    inputname = sys.argv[1].split('.')[0]
    order = make_order(label)
    print(order)
    write_file(label, inputname, n, order)

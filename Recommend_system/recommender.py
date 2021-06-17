import pandas as pd
import numpy as np
import sys
import time


# Matrix factorization을 하는 함수
def matrix_factorization(R, P, Q):
    Epochs = 100
    alpha = 0.005
    beta = 0.02
    Q = Q.T
    for epoch in range(Epochs):
        if epoch == 60:
            alpha = 0.0005
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # update P, Q
                    eij = R[i][j] - (np.dot(P[i, :], Q[:, j])+beta)
                    P[i, :] += alpha * (2 * eij * Q[:, j] - beta * P[i, :])
                    Q[:, j] += alpha * (2 * eij * P[i, :] - beta * Q[:, j])
        newR = np.dot(P, Q)+beta
        cost = 0
        cnt = 0
        # calculate cost
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    cost += (R[i][j] - newR[i][j])**2
                    cnt += 1
        cost = (cost/cnt)**0.5
        print(epoch, cost)
        # cost가 0.1보다 작으면 학습 종료
        if cost < 0.1:
            break
    return np.dot(P, Q)


# pre_preference를 예측하기 위해 MF를 함
def matrix_factorization_pre(R, P, Q):
    Epochs = 80
    alpha = 0.003
    beta = 0.01
    Q = Q.T
    for epoch in range(Epochs):
        if epoch == 50:
            alpha = 0.0005
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - (np.dot(P[i, :], Q[:, j])+beta)
                    P[i, :] += alpha * (2 * eij * Q[:, j] - beta * P[i, :])
                    Q[:, j] += alpha * (2 * eij * P[i, :] - beta * Q[:, j])
        newR = np.dot(P, Q)+beta
        cost = 0
        cnt = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    cost += (R[i][j] - newR[i][j])**2
                    cnt += 1
        cost = (cost/cnt)**0.5
        print(epoch, cost)
        if cost < 0.001:
            break
    return np.dot(P, Q)


# 최종 ouput file에 들어갈 column과 index를 결정함
def get_new_column_and_index(base_df, test_df):
    mt = test_df.max().values
    mf = base_df.max().values
    user_max = max(mt[0], mf[0])
    item_max = max(mt[1], mf[1])
    column_list = range(1, item_max+1)
    index_list = range(1, user_max+1)
    return column_list, index_list


# pre-preference가 낮으면 R에 1을 넣음
def insert_pre_preference(R, N, M, K):
    toOne = np.array([np.array([1 if x > 0 else 0 for x in row]) for row in R])
    P2 = np.random.rand(N, K)*0.01+0.1
    Q2 = np.random.rand(M, K)*0.01+0.1
    # MF로 pre-preference를 구함
    pre = matrix_factorization_pre(toOne, P2, Q2)
    Q1 = np.percentile(pre, 5)
    # preferencr가 하위 5%면 R에 1을 삽입함
    for i in range(len(pre)):
        for j in range(len(pre[i])):
            if pre[i][j] < Q1 and R[i][j] == 0:
                R[i][j] = 1


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            "USAGE: python recommender.py <base filename> <test filename>")
        sys.exit()
    basefile = f'./data/{sys.argv[1]}'
    testfile = f'./data/{sys.argv[2]}'
    base_df = pd.read_csv(basefile, sep='\t', header=None, names=[
        'user_id', 'item_id', 'rating', 'time_stamp'])
    base_df.drop('time_stamp', axis=1, inplace=True)
    test_df = pd.read_csv(testfile, sep='\t', header=None, names=[
        'user_id', 'item_id', 'rating', 'time_stamp'])
    test_df.drop('time_stamp', axis=1, inplace=True)
    rating_matrix = base_df.pivot(
        values='rating', index='user_id', columns='item_id').fillna(0)
    # 원래 갖고있던 column과 index
    column_list = rating_matrix.columns
    index_list = rating_matrix.index
    R = rating_matrix.values
    # output file에 쓰일 column과 index
    res_column_list, res_index_list = get_new_column_and_index(
        base_df, test_df)
    # number of Users
    N = len(R)
    # number of Movies
    M = len(R[0])
    # number of Features
    K = 3
    # pre-preference가 낮은면 R에 1을 삽입함
    insert_pre_preference(R, N, M, K)
    start_time = time.time()
    P = np.random.rand(N, K)*0.01+1
    Q = np.random.rand(M, K)*0.01+1
    predict_R = matrix_factorization(R, P, Q)
    # 값이 너무 작거나 크게 예측된걸 보정함
    for i in range(len(predict_R)):
        for j in range(len(predict_R[i])):
            if predict_R[i][j] < 1:
                predict_R[i][j] = 1
            if predict_R[i][j] > 5:
                predict_R[i][j] = 5
    res_mean = np.mean(predict_R)
    R_df = pd.DataFrame(predict_R, index=index_list, columns=column_list)
    # 미리 저장해뒀던 index, column으로 reindex함
    # 모르는 rating은 전체의 평균으로 설정함
    result = R_df.reindex(index=res_index_list,
                          columns=res_column_list).fillna(res_mean)
    predict = result.stack(dropna=False)
    print(f'{time.time()-start_time}s')
    output_filename = f'./test/{sys.argv[1]}_prediction.txt'
    predict.to_csv(output_filename, sep='\t', index=True, header=False)

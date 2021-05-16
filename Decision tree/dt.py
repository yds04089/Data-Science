import pandas as pd
import numpy as np
import sys


# calculate entropy
def entropy(table, label_val, total_small, attr):
    info = 0
    for j in label_val:
        cnt = table[attr][j]
        pi = cnt/total_small
        if pi:
            info -= pi*np.log2(pi)
    return info


# select test attribute
def select_attribute(df, attributes, label_val, info_d, label):
    selected_attr = ""
    selected_gain = 0
    for index in attributes:
        total = df[index].count()
        attributes_idx = df[index].unique()
        info = 0
        # 값이 없으면 0으로 채움
        table = df.groupby([index, label]).size().unstack(fill_value=0).stack()
        # attribute마다 information gain을 구하고 가장 큰 것을 test attribute로 설정
        for attr in attributes_idx:
            total_small = table[attr].sum()
            info_ = entropy(table, label_val, total_small, attr)
            info += (total_small/total)*info_
        information_gain = info_d - info

        if selected_gain < information_gain:
            selected_attr = index
            selected_gain = information_gain
    return selected_attr, selected_gain


# majority voting
def majority_vote(table, label, major_dict):
    majority = table[label].value_counts(sort=True)
    major_num = majority[0]
    final = majority.index[0]
    cnt = major_dict[final]
    for i in range(len(majority)):
        # majority 개수가 같은 것들이 있을 때에는 전체 데이터에서 가장 수가 적은 label 선택
        if majority[i] == major_num:
            if cnt > major_dict[majority.index[i]]:
                final = majority.index[i]
                cnt = major_dict[final]
    return final


# build decision tree
def decision_tree(table, attributes, depth, major_dict):
    label = table.columns[-1]
    label_val = table[label].unique()
    total = table[label].count()
    attr_cnt = len(attributes)
    # 해당 노드의 데이터들이 하나의 class label을 갖게된 경우
    if len(label_val) == 1:
        return label_val[0]
    # branch로 쪼갤 attribute가 없는 경우 -> majority vote
    if attr_cnt == 1:
        return majority_vote(table, label, major_dict)
    # 해당 노드에 데이터 수가 두개 이하인 경우 -> majority vote (pre-pruning)
    if total <= 2:
        return majority_vote(table, label, major_dict)
    # 해당 node의 전체 데이터 entropy 계산
    info_d = 0
    for i in table[label].value_counts():
        pi = i/total
        info_d -= pi*np.log2(pi)
    # test attribute를 선택
    selected_attr, selected_gain = select_attribute(
        table, attributes, label_val, info_d, label)
    # information gain이 0.1보다 작고 tree의 깊이가 1보다 깊은 경우 -> majority vote (pre-pruning)
    if selected_gain < 0.1 and depth > 1:
        return majority_vote(table, label, major_dict)
    tree = {selected_attr: {}}
    # test attribute를 제외한 나머지 attribute들
    not_selected_attrs = np.delete(
        attributes, np.where(attributes == selected_attr))
    # test attribute를 기준으로 branch를 내리도록 decision_tree()를 재귀적으로 호출
    for i in table[selected_attr].unique():
        selected_table = table[table[selected_attr] == i]
        branch = decision_tree(
            selected_table, not_selected_attrs, depth+1, major_dict)
        tree[selected_attr][i] = branch
    return(tree)


# 생성된 tree를 이용해 test data를 classify함
def classify(tree, data, major_result):
    while True:
        # tree가 하나의 class value로만 이루어진 경우
        if type(tree) == str:
            return major_result
        attr = list(tree.keys())[0]
        if attr in data:
            if data[attr] in tree[attr]:
                # leaf node이면 그 value를 리턴
                if type(tree[attr][data[attr]]) == str:
                    return tree[attr][data[attr]]
                tree = tree[attr][data[attr]]
            # test data의 attribute value가 존재하지 않는 경우 전체 dataset의 majority vote결과를 리턴
            else:
                return major_result
        # data에 없는 attribute가 tree의 test attribute인 경우  -> 전체 data에서의 major voting값을 리턴함
        else:
            return major_result


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "USAGE: python dt.py <training filename> <test filename> <output filename>")
        sys.exit()
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    df = pd.read_csv(f'./data/{train_file}', sep='\t')
    dt = pd.read_csv(f'./data/{test_file}', sep='\t')
    # class label
    label = df.columns[-1]
    # class label values
    label_val = df[label].unique()
    total = df[label].count()
    # 전체 dataset에서 각 class label을 갖는 데이터 수 저장
    majority_table = df[label].value_counts(sort=True)
    major_dict = {}
    for i in range(len(majority_table)):
        major_dict[majority_table.index[i]] = majority_table[i]
    major_result = majority_vote(df, label, major_dict)
    attributes = df.columns[:-1]
    # build decision tree
    tree = decision_tree(df, attributes, 1, major_dict)
    test_total = len(dt)
    res = []
    # test dataset을 한줄씩 decision tree에 넣어서 classify함
    for i in range(test_total):
        test = dt.iloc[i]
        # test data를 dictionary 형태로 재구성
        test_data = {}
        for k in range(len(test)):
            test_data[test.index[k]] = test[k]
        # decision tree에 적용
        ans = classify(tree, test_data, major_result)
        res.append(ans)
    # test dataset에 class label을 추가
    dt.loc[:, label] = res
    print(dt)
    # result 출력
    dt.to_csv(f'./test/{output_file}', sep="\t")

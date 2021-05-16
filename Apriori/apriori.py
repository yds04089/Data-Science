from itertools import combinations
import sys


# leng 길이의 candidates 생성
def make_candidate(Lk, leng):
    ret = set([])
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)):
            can = Lk[i] | Lk[j]
            if len(can) == leng:
                # pruning
                if leng > 2:
                    can_com = list(combinations(can, leng-1))
                    check_prun = set([])
                    for item in can_com:
                        check_prun.add(frozenset(item))
                    check_prun = frozenset(check_prun)
                    if check_prun.issubset(Lk):
                        ret.add(frozenset(can))
                else:
                    ret.add(frozenset(can))
    return ret


# input file을 scan하면서 candidate들의 개수를 count
def scan_db(f, Ck, freq_dict):
    f.seek(0)
    cnt = {}
    while True:
        line = f.readline()
        if not line:
            break
        line = [int(i) for i in line.split()]
        # transaction마다 candidate들을 모두 확인하고 개수 count함
        for can in Ck:
            if can.issubset(line):
                if frozenset(can) in cnt:
                    freq_dict[frozenset(can)] += 1
                    cnt[frozenset(can)] += 1
                else:
                    freq_dict[frozenset(can)] = 1
                    cnt[frozenset(can)] = 1
    return cnt


# candidate들 중 frequent pattern인 것만 찾아냄
def find_freq(Ck, min_sup, freq, freq_dict):
    ret = []
    for item in Ck:
        if Ck[item] >= min_sup:
            ret.append(item)
            freq.add(item)
        else:
            del(freq_dict[item])
    return ret


# association rule에 따라 frequent pattern들 사이의 support, confidence 찾음
def association(freq, freq_dict, total):
    freq = list(freq)
    res = ""
    for items in freq:
        items = list(items)
        sup_cnt = freq_dict[frozenset(items)]
        sup = format(round(sup_cnt/total*100, 2), ".2f")
        size = len(items)
        for i in range(1, size):
            comb = list(combinations(items, i))
            for X in comb:
                X = frozenset(X)
                Y = frozenset(items) - X
                conf_X = format(round(sup_cnt/freq_dict[X]*100, 2), ".2f")
                res += f'{set(X)}\t{set(Y)}\t{sup}\t{conf_X}\n'
    return res


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            "USAGE: python apriori.py <minimum support> <input filename> <output filename>")
        sys.exit()
    min_support = int(sys.argv[1])
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    # check minimum support
    if min_support <= 0:
        print("minimum support must be a possitive.")
        sys.exit()
    f_in = open(f'./{input_file}', 'r')
    # freq pattern
    freq = set([])
    # count freq pattern
    freq_dict = {}
    # 길이가 1인 candidate 개수 세는 dictionary
    can_1 = {}
    # transaction의 총 개수
    total_cnt = 0
    while True:
        line = f_in.readline()
        if not line:
            break
        total_cnt += 1

        # 문자열을 int로 변환
        line = [int(i) for i in line.split()]
        for i in line:
            if i in can_1:
                can_1[i] += 1
            else:
                can_1[i] = 1
    min_sup = total_cnt*min_support/100
    # 길이가 1인 freq patterns
    L1 = []
    for i in can_1:
        if can_1[i] >= min_sup:
            L1.append(set([i]))
            freq_dict[frozenset([i])] = can_1[i]

    leng = 2
    L_now = L1
    while True:
        Ck = make_candidate(L_now, leng)
        Ck_cnt = scan_db(f_in, Ck, freq_dict)
        L_next = find_freq(Ck_cnt, min_sup, freq, freq_dict)

        if not L_next:
            break
        L_now = L_next
        leng += 1
    f_in.close()

    # association rule
    result = association(freq, freq_dict, total_cnt)
    print(len(freq))
    f_out = open(f'./{output_file}', 'w')
    f_out.write(result)
    f_out.close()

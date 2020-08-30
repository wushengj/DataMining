'''
Python command: $ python3 bfr.py <input_path> <n_cluster> <out_file1_json> <out_file2_csv>

'''

import sys
import json
import csv
from collections import OrderedDict, defaultdict
import os
from os import listdir
import time
import random
from math import sqrt, ceil, floor

from pyspark import SparkContext, SparkConf, StorageLevel

SAMPLE_RATE = 0.1
RANDOM = 6836
# K = 10
FIRSTKTIMES = 3
RESTKTIMES = 3
MAXITER = 16
ALPHA = 4 # 3 or 2
CONVERGENCE_DIS = 1e-5
INFINITY = 9999999999999999999999999999999999999999999999999999999999999999999999999

input_file = sys.argv[1]
cluster_cnt = int(sys.argv[2])
output_file_cluster = sys.argv[3]
output_file_intermediate = sys.argv[4]

K = cluster_cnt

conf = SparkConf() \
    .setAppName("INF553") \
    .setMaster('local[*]')


def getFilePaths(path):
    files = listdir(path.replace('file://', ''))
    path_l = [path + '/' + f for f in files]
    path_l.sort()
    firstp = path_l[0]
    restp_l = path_l[1:]
    return firstp, restp_l


def readTxt(str):
    '''
    read each line
    :param str: "num, num, ..."
    :return: (i, [d,...])
    '''
    str.strip()
    elems = str.split(',')
    key = int(elems[0])
    value = [float(elems[i]) for i in range(1, len(elems))]
    return (key, value)


def getRestByKey(whole_l, ex_l):
    '''
    MAY OPTIMIZE LATER
    get the rest of the data in chunk, exluded from ex_l
    :param whole_l: [(i, [d,...]), (i, [d,...]), ...]
    :param ex_l: [(i, [d,...]), ...]
    :return: [(i, [d,...]), ...]
    '''
    exi_l = [j[0] for j in ex_l]
    return [i for i in whole_l if i[0] not in exi_l]


def getRestByValue(whole_l, ex_l):
    '''
    MAY OPTIMIZE LATER
    get the rest of the data in chunk, exluded from ex_l
    :param whole_l: [(i, [d,...]), (i, [d,...]), ...]
    :param ex_l: [(i, [d,...]), ...]
    :return: [(i, [d,...]), ...]
    '''
    exi_l = [j[1] for j in ex_l]
    # print('whole_l')
    # print(whole_l)
    # print('ex_l')
    # print(ex_l)
    return [i for i in whole_l if i[1] not in exi_l]


def getDistance(p, tar):
    '''
    get a list of distance from target tar to each of the data point in centroid list c_l
    :param p: [d1, d2,...]
    :param tar: [d1, d2,...]
    :return: dis
    '''
    dis = 0
    for i in range(len(p)):
        dis += (p[i] - tar[i]) ** 2
    return sqrt(dis)


def addList(l1, l2):
    '''
    get a list of sum of elements from l1 and l2, l1 l2 have the same length
    :param l1: [d,...]
    :param l2: [d,...]
    :return: [d,...]
    '''
    d = len(l1)
    add_l = [l1[i] + l2[i] for i in range(d)]
    return add_l


def getCentroid(cv_l):
    '''
    get the centroid of the dataset
    :param cv_l: [[d,...], [d,...], ...]
    :return: [d,...]
    '''
    dimension = len(cv_l[0])
    size = len(cv_l)
    c = [0] * dimension
    for v_l  in cv_l:
        c = addList(c, v_l)
    c = [add / size for add in c]
    return c


def nextCentroid(l, dis_d, new_c_id, new_c_value):
    '''
    find the farthest point in l using the distance from the point to the centroids (the distance from it to the nearest centroid)
    :param l: [(i, [d,...]), (i, [d,...]), ...]
    :param dis_d: {i: (i, dis), ...}
    :param max_dis: float
    :param new_c_id: i
    :param new_c_value: [d,...]
    :return: new_dis_d, max_dis, foundedc_id, foundedc_value
    '''
    # print('new_c_value',new_c_value)
    new_dis_d = dis_d
    max_dis = 0
    this_foundedc_id = None
    this_foundedc_value = None
    # print('new_c_value:', new_c_value)
    for p in l:
        id = p[0]
        value = p[1]
        if type(value) == int:
            print('point id:', id, 'has', value)
        if type(new_c_value) == int:
            print('centroid point id:', new_c_id, 'has', new_c_value)
        # print(new_c_value)
        # print(value)
        if id == new_c_id:
            new_dis_d[id] = (new_c_id, 0)
        else:
            pc_dis = getDistance(value, new_c_value)
            if id not in dis_d.keys():
                print('Warn! point id not in dis_d')
            old_dis = dis_d[id][1]
            if pc_dis < old_dis:
                new_dis_d[id] = (new_c_id, pc_dis)
            if new_dis_d[id][1] > max_dis:
                max_dis = new_dis_d[id][1]
                this_foundedc_id = id
                this_foundedc_value = value
    return new_dis_d, this_foundedc_id, this_foundedc_value


def selectKCentroids(l, n_clusters):
    '''
    iteratively select disperse centroids
    :param l: [(i, [d,...]), (i, [d,...]), ...]
    :param n_clusters: int
    :return: [(ci, [d,...]), (ci, [d,...]), ...]
    '''
    random.seed(RANDOM)
    first_centroid_key = random.randint(0, len(l)-1)
    first_centroid_value = l[first_centroid_key][1]
    kth = 1
    ci_l = [(first_centroid_key, first_centroid_value)]
    dis_d = {}
    max_dis = 0
    for p in l:
        id = p[0]
        value = p[1]
        # if type(value) == int:
        #     print('point id:', id, 'has', value)
        if id == first_centroid_key:
            dis_d[id] = (first_centroid_key, 0)
        else:
            pc_dis = getDistance(value, first_centroid_value)
            if pc_dis > max_dis:
                max_dis = pc_dis
                foundedc_id = id
                foundedc_value = value
                # if type(foundedc_value) == int:
                #     print('founded point id:', id, 'has', value)
            dis_d[id] = (first_centroid_key, pc_dis)
    # print(foundedc_id)
    # print(foundedc_value)
    while kth < n_clusters:
        # if kth == 3:
        #     print(dis_d)
        #     print(max_dis)
        #     print(foundedc_id)
        #     print(foundedc_value)
        kth += 1
        ci_l.append((foundedc_id, foundedc_value))
        # print('################# Iteration no.%d' % kth)
        # if type(foundedc_value) == int:
        #     print('c to calculate id:', foundedc_id, 'has', foundedc_value)
        new_dis_d, new_foundedc_id, new_foundedc_value = nextCentroid(l, dis_d, foundedc_id, foundedc_value)
        dis_d, foundedc_id, foundedc_value = new_dis_d, new_foundedc_id, new_foundedc_value
    return ci_l


def nearestPoint(p, point_l):
    '''
    get the nearest point of p in point_l
    :param p: (i, [d,...])
    :param point_l: [(i, [d,...]), (i, [d,...]), ...] / [(ci, [d,...]), (ci, [d,...]), ...]
    :return: (1, [d1, d2,...])
    '''
    pdis_l = []
    for tar in point_l:
        dis = getDistance(p[1], tar[1])
        pdis_l.append((tar, dis))
    min_dis = pdis_l[0][1]
    min_dis_p = pdis_l[0][0]
    for t in pdis_l:
        if t[1] < min_dis:
            min_dis = t[1]
            min_dis_p = t[0]
    return min_dis_p


def reassignCentroids(l, ci_l):
    '''
    assign each p in l to a centroid in ci_l
    :param l: [(i, [d,...]), (i, [d,...]), ...]
    :param ci_l: [(ci, [d,...]), (ci, [d,...]), ...]
    :return point with its centroid list cp_l: [(ci, (i, [d,...])), (ci, (i, [d,...])), ...]
    :return centroid list ci_l: [(ci, [d,...]), (ci, [d,...]), ...]
    '''
    cp_d = defaultdict(list)
    for p in l:
        near = nearestPoint(p, ci_l)
        c = near[0]
        cp_d[c].append(p)
    cp_l = []
    ci_l = []
    for c, p_l in cp_d.items():
        d_l = []
        for p in p_l:
            cp_l.append((c, p))
            d_l.append(p[1])
        cv = getCentroid(d_l)
        ci_l.append((c, cv))

    return cp_l, ci_l


def decideKMIter(c_l, convergence_dis=CONVERGENCE_DIS):
    '''
    only if convergence, iteration will not happen
    :param c_l: [[(ci, [d,...]), (ci, [d,...]), ...], ... ]
    :param max_iter: int
    :return: True / False
    '''
    ci_l_a = c_l[-2]
    ci_l_b = c_l[-1]
    d_a = [t[1] for t in ci_l_a]
    d_b = [t[1] for t in ci_l_b]
    if len(d_a) > len(d_b):
        print('DS cluster number reduced from', len(d_a), 'to', len(d_b))
        return True
    d_num = len(d_a)
    # print('d_num:', d_num)
    dis_ab_sum = sum([getDistance(d_a[i], d_b[i]) for i in range(d_num)])
    print('dis_ab_sum:', dis_ab_sum)
    if dis_ab_sum > convergence_dis:
        return True
    else:
        return False


def reassignPoints(p, ci_l):
    '''
    assign p to a centroid in ci_l
    :param p: (i, [d,...])
    :param ci_l: [(ci, [d,...]), (ci, [d,...]), ...]
    :return point with its centroid cp: (ci, (i, [d,...])
    '''
    near = nearestPoint(p, ci_l)
    c = near[0]
    return (c, p)


def KMeans(l, n_clusters, max_iter):
    '''
    Spark
    :param l: [(i, [d,...]), (i, [d,...]), ...]
    :param n_clusters: int
    :param max_iter: int
    :return: [(c, (i, [d,...])), (c, (i, [d,...])), ...]
    '''
    # RDD = sc.parallelize(l).map()
    # if l:
    l_RDD = sc.parallelize(l).persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
    # c_l = []
    ci_l_n = selectKCentroids(l, n_clusters)
    # c_l.append(ci_l)
    ci_l_o = None
    keepIter = True
    iter_num = 1
    while keepIter and iter_num < max_iter:
        ci_l_o = [x for x in ci_l_n]
        cp_l = l_RDD.map(lambda p: reassignPoints(p, ci_l_o))
        ci_l_n = cp_l.groupByKey().mapValues(lambda l: [t[1] for t in l]).mapValues(getCentroid).collect()
        # c_l.append(ci_l)
        keepIter = decideKMIter([ci_l_o, ci_l_n])
        iter_num += 1
        print('Iter #%d' % iter_num)
    return_cp_l = cp_l.collect()
    return return_cp_l
    # else:
    #     return []


def calculateSigma(info):
    '''
    a vector of standard deviation of every dth dimention
    :param info: (N, SUM, SUMSQ)
    :return: [sgm,...]
    '''
    # size = len(d_l)
    n = info[0]
    sum = info[1]
    sumsq = info[2]
    d_num = len(sum)
    sgm_l = []
    i = 0
    while i < d_num:
        sgm = sqrt(sumsq[i] / n - (sum[i] / n) ** 2)
        sgm_l.append(sgm)
        i += 1
    return sgm_l


def summarize(l):
    '''
    represent DS / CS by stats
    :param l: [(i1, [d,...]), (i4, [d,...]), ...]
    :return: (N, SUM, SUMSQ)
    '''
    N = len(l)
    d_num = len(l[0][1])
    SUM = [0] * d_num
    SUMSQ = [0] * d_num
    for p in l:
        for i in range(d_num):
            SUM[i] += p[1][i]
            SUMSQ[i] += p[1][i] ** 2
    # d_l = [p[1] for p in l]
    # C = getCentroid(d_l)
    # SGM = calculateSigma(d_l, d_num, N, SUM, SUMSQ)
    return (N, SUM, SUMSQ)


def Mahalanobis(p, c, sgm):
    '''
    calculate Mahalanobis Distance of point p and centroid c
    :param p: (i, [d,...])
    :param c: [d,...]
    :param sgm: [d,...]
    :return: dis
    '''
    d = len(c)
    dis = 0
    for i in range(d):
        if sgm[i] == 0:
            print(sgm)
        dis += ((p[1][i] - c[i]) / sgm[i]) ** 2
    return sqrt(dis)


def findNearestDSorCS(p, sum_l, large=INFINITY):
    '''
    find the nearest DS or CS to p by calculating Mahalanobis Distance
    :param p: (i, [d,...])
    :param sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :param d: int
    :return: (c, dis)
    '''
    c = large
    dis = large
    for set in sum_l:
        n = set[1][0]
        centroid = [sum_i / n for sum_i in set[1][1]]
        sgm = calculateSigma(set[1])
        M_dis = Mahalanobis(p, centroid, sgm)
        if M_dis < dis:
            c = set[0]
            dis = M_dis
    return (c, dis)


def assignSet(p, DS_sum, CS_sum, alpha = ALPHA):
    '''
    assign new point p to a set
    :param p: (i1, [d,...])
    :param DS_sum: [(c, (N, SUM, SUMSQ)), ...]
    :param CS_sum: [(c, (N, SUM, SUMSQ)), ...]
    :param RS_list: [(i, [d,...]), ...]
    :return: (i1, ([d,...], [DS], [CS], [RS]))
    '''
    d = len(p[1])
    threshold = alpha * sqrt(d)
    nearest_DS = findNearestDSorCS(p, DS_sum)
    if nearest_DS[1] < threshold:
        assignedP = (p[0], (p[1], [nearest_DS[0]], [], []))
    else:
        nearest_CS = findNearestDSorCS(p, CS_sum)
        if nearest_CS[1] < threshold:
            assignedP = (p[0], (p[1], [], [nearest_CS[0]], []))
        else:
            assignedP = (p[0], (p[1], [], [], [1]))
    return assignedP


def updateDSorCS(cp_l, sum_l):
    '''

    :param cp_l: (c, [(i, [d,...]), ...])
    :param sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :return: (c, (N, SUM, SUMSQ))
    '''
    c = cp_l[0]
    for c_info in sum_l:
        if c_info[0] == c:
            old = c_info[1]
    add_info = summarize(cp_l[1])
    # new = tuple([add_info[i] + old[i] for i in range(3)])
    new = [add_info[0] + old[0]]
    for i in range(1, 3):
        add = addList(add_info[i], old[i])
        new.append(add)
    return (c, tuple(new))


def lonelyDS(DSsum_l):
    '''
    chech if there exists at least one cluster that has only one point within it
    :param DSsum_l: [(c, (N, SUM, SUMSQ)), ...]
    :return: boolean
    '''
    for c_info in DS_sum:
        if c_info[1][0] == 1:
            print('lonely DS.')
            return True
    return False


def combineCS(sum_l_first, ith, sum_l_second, jth, combine_on, how):
    '''
    both - combine sum_l_first's ith elem and sum_l_second's jth elem, merged elem goes to sum_l_first
    :param sum_l_a: [(c, (N, SUM, SUMSQ)), ...]
    :param ith: int
    :param sum_l_b: [(c, (N, SUM, SUMSQ)), ...]
    :param jth: int
    :param combine_on: 'first' / 'second' / 'both'
    :return new_sum_l_a: [(c, (N, SUM, SUMSQ)), ...]
    :return new_sum_l_b: [(c, (N, SUM, SUMSQ)), ...]
    :return changed_c: (old_c, new_c)
    '''
    if combine_on == 'first':
        a_elem = sum_l_first[ith]
        b_elem = sum_l_first[jth]
    elif combine_on == 'second':
        a_elem = sum_l_second[ith]
        b_elem = sum_l_second[jth]
    elif combine_on == 'both':
        a_elem = sum_l_first[ith]
        b_elem = sum_l_second[jth]
    else:
        print('wrong')

    if how == 'CSCS':
        if a_elem[0] > b_elem[0]:
            id = a_elem[0]
            old_c = b_elem[0]
            new_c = a_elem[0]
        elif a_elem[0] < b_elem[0]:
            id = b_elem[0]
            old_c = a_elem[0]
            new_c = b_elem[0]
        else:
            print('Error: Merging clusters have same ID.')
    elif how == 'DSCS':
        id = a_elem[0]
        old_c = b_elem[0]
        new_c = a_elem[0]
    merged_elem = (id, (a_elem[1][0] + b_elem[1][0], addList(a_elem[1][1], b_elem[1][1]), addList(a_elem[1][2], b_elem[1][2])))
    changed_c = (old_c, new_c)

    if combine_on == 'first':
        sum_l_first.remove(a_elem)
        sum_l_first.remove(b_elem)
        sum_l_first += [merged_elem]
        # sum_l_first = [elem for elem in sum_l_first if elem != a_elem and elem != b_elem]
    elif combine_on == 'second':
        sum_l_second.remove(a_elem)
        sum_l_second.remove(b_elem)
        sum_l_second += [merged_elem]
    elif combine_on == 'both':
        # sum_l_first.remove(a_elem)
        # sum_l_second.remove(b_elem)
        # sum_l_second += [merged_elem]
        sum_l_first = [elem for elem in sum_l_first if elem != a_elem]
        sum_l_second = [elem for elem in sum_l_second if elem != b_elem]
        sum_l_first += [merged_elem]
    else:
        print('Error: Wrong merge method (combine_on).')

    return sum_l_first, sum_l_second, changed_c


def decideMerge(elem_i, elem_j, alpha=ALPHA):
    '''

    :param elem_i: (c, (N, SUM, SUMSQ))
    :param elem_j: (c, (N, SUM, SUMSQ))
    :return: boolean
    '''
    d = len(elem_i[1][1])
    threshold = alpha * sqrt(d)
    if elem_i[1][0] < elem_j[1][0]:
        elem_fewer = elem_i
        elem_more = elem_j
    else:
        elem_fewer = elem_j
        elem_more = elem_i

    point = (elem_fewer[0], [sum_i / elem_fewer[1][0] for sum_i in elem_fewer[1][1]])
    cen = [sum_j / elem_more[1][0] for sum_j in elem_more[1][1]]
    sgm = calculateSigma(elem_more[1])
    M_dis = Mahalanobis(point, cen, sgm)
    if M_dis < threshold:
        return True
    else:
        return False


def decideMergeIter(sum_l_a, sum_l_b, how):
    '''

    :param sum_l_a: [(c, (N, SUM, SUMSQ)), ...]
    :param sum_l_b: [(c, (N, SUM, SUMSQ)), ...]
    :param how: 'CSCS' / 'DSCS'
    :return: 'True', combine_on, i, j / 'False', 'nothing', 'nothing', 'nothing'
    '''
    # check if elems in sum_l_a cannot merge
    if how == 'CSCS':
        for i in range(len(sum_l_a)):
            for j in range(i+1, len(sum_l_a)):
                if decideMerge(sum_l_a[i], sum_l_a[j]):
                    return 'True', 'first', i, j
    # check if elems in sum_l_b cannot merge
    # if how == 'CSCS' or 'DSCS':
    for i in range(len(sum_l_b)):
        for j in range(i + 1, len(sum_l_b)):
            if decideMerge(sum_l_b[i], sum_l_b[j]):
                return 'True', 'second', i, j
    # check if elems in sum_l_a and sum_l_b cannot merge
    # if how == 'CSCS' or 'DSCS':
    for i in range(len(sum_l_a)):
        for j in range(len(sum_l_b)):
            if decideMerge(sum_l_a[i], sum_l_b[j]):
                return 'True', 'both', i, j
    return 'False', 'nothing', 'nothing', 'nothing'


def mergeCS(old_sum_l, new_sum_l, how, alpha=ALPHA):
    '''
    iteratively merge smaller CS to larger CS
    :param sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :param new_sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :param how: 'CSCS' / 'DSCS'
    :return old_sum_l, new_sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :return change_l: [(old_c, new_c), ...]
    '''
    change_l = []
    keep_iter, on, ith, jth = decideMergeIter(old_sum_l, new_sum_l, how)
    while keep_iter == 'True':
        new_old_sum_l, new_new_sum_l, change = combineCS(old_sum_l, ith, new_sum_l, jth, on, how)
        old_sum_l, new_sum_l = new_old_sum_l, new_new_sum_l
        change_l.append(change)
        keep_iter, on, ith, jth = decideMergeIter(old_sum_l, new_sum_l, how)

    if how == 'CSCS':
        merged_sum_l = old_sum_l + new_sum_l
        return merged_sum_l, change_l
    else:
        return old_sum_l, new_sum_l, change_l


def intermediate(rd_num, sum_l_ds, sum_l_cs, l_rs):
    '''

    :param rd_num: int
    :param sum_l_ds: [(c, (N, SUM, SUMSQ)), ...]
    :param sum_l_cs: [(c, (N, SUM, SUMSQ)), ...]
    :param l_rs: [(i, [d,...]), ...]
    :return: (rd_num, ds_num, ds_p_num, cs_num, cs_p_num, rs_p_num)
    '''
    ds_num = 0
    ds_p_num = 0
    for elem in sum_l_ds:
        ds_num += 1
        ds_p_num += elem[1][0]
    cs_num = 0
    cs_p_num = 0
    for elem in sum_l_cs:
        cs_num += 1
        cs_p_num += elem[1][0]
    rs_p_num = len(l_rs)
    return (rd_num, ds_num, ds_p_num, cs_num, cs_p_num, rs_p_num)


def mapCluster(p_info):
    '''

    :param p_info: (i1, ([d,...], [DS], [CS], [RS]))
    :return: (i, c)
    '''
    if p_info[1][1]:
        return (p_info[0], p_info[1][1][0])
    elif p_info[1][2]:
        return (p_info[0], p_info[1][2][0])
    else:
        return (p_info[0], -1)


def updateClusterResult_RS(cluster_result_d, update_l):
    '''

    :param cluster_result_d: {i: c, ...}
    :param update_l: [(i, new_c), ...]
    :return: {i: c, ...}
    '''
    for update_info in update_l:
        cluster_result_d[update_info[0]] = update_info[1]
    return cluster_result_d


def updateClusterResult_CS(cluster_result_d, update_l):
    '''

    :param cluster_result_d: {i: c, ...}
    :param update_l: [(old_c, new_c), ...]
    :return: {i: c, ...}
    '''
    for update_info in update_l:
        for idn, c in cluster_result_d.items():
            if c == update_info[0]:
                cluster_result_d[idn] = update_info[1]
    return cluster_result_d


def checkLostCluster(old_sum_l, new_sum_l):
    '''

    :param old_sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :param new_sum_l: [(c, (N, SUM, SUMSQ)), ...]
    :return: [(c, (N, SUM, SUMSQ)), ...]
    '''
    d = len(old_sum_l[0][1][1])
    new_c_l = [c_info[0] for c_info in new_sum_l]
    # old_c_l = list(set([c_info[0] for c_info in DS_sum]))
    lost_sum_l = [c_info for c_info in old_sum_l if c_info[0] not in new_c_l]
    if lost_sum_l:
        for c_info in lost_sum_l:
            new_sum_l.append(c_info)
    return new_sum_l


if __name__ == "__main__":
    start_time = time.time()

    sc = SparkContext(conf=conf)

    output_a_dict = {}
    output_b_list = []

    firstPath, restPaths = getFilePaths(input_file)
    total_file_num = len(restPaths) + 1
    print(firstPath)
    # firstPath = 'file:///Users/shengjiawu/Desktop/INF553/HW/HW5/hw5_data/test1/data0.txt'
    # first chunk
    firstChunk = sc.textFile(firstPath).map(readTxt).collect()
    load_chunk_num = 1
    ## split first chunk to sample and the rest
    DS_sum = [('c', (1, 'SUM', 'SUMSQ'))]
    while lonelyDS(DS_sum) or len(DS_sum) < K:
        RANDOM += 1
        random.seed(RANDOM)
        sample_list = random.sample(firstChunk, ceil(len(firstChunk) * SAMPLE_RATE))
        rest_start_time = time.time()
        rest_list = getRestByKey(firstChunk, sample_list)
        # print('get rest time:', time.time() - rest_start_time)
        ## KMeans on sample to get DS
        DSKMeans_start_time = time.time()                                                                                   # TIME!!!!!!!!!!
        DS_list = KMeans(sample_list, K, MAXITER) # [(c, (i, [d,...])), (c, (i, [d,...])), ...]
        DSKMeans_end_time = time.time()                                                                                     # TIME!!!!!!!!!!
        print('first load DSKMeans time:', DSKMeans_end_time - DSKMeans_start_time)                                         # TIME!!!!!!!!!!
        DS_sum = sc.parallelize(DS_list) \
            .groupByKey() \
            .map(lambda x: (x[0], list(x[1]))) \
            .mapValues(summarize) \
            .collect()                                             # [(c, (N, SUM, SUMSQ)), ...]
        print('length of DS_sum:', len(DS_sum))

    # KMeans on the rest to get CS, RS
    print("start to get 1st cs rs")
    mixCSRSKMeans_start_time = time.time()                                                                              # TIME!!!!!!!!!!
    mixCSRS_list = KMeans(rest_list, FIRSTKTIMES * K, MAXITER) # [(c, (i, [d,...])),...]
    # print('mixCSRS_list', mixCSRS_list)
    mixCSRSKMeans_end_time = time.time()                                                                                # TIME!!!!!!!!!!
    print('first load mixCSRSKMeans time:', mixCSRSKMeans_end_time - mixCSRSKMeans_start_time)                          # TIME!!!!!!!!!!

    sepCSRSRDD = sc.parallelize(mixCSRS_list) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1])))\
        .persist(storageLevel=StorageLevel.MEMORY_AND_DISK)
    # print('sepCSRSRDD')
    # print(sepCSRSRDD.collect())
    # [(c, [(i, [d, ...]), ...]), ...]
    CS_RDD = sepCSRSRDD.filter(lambda x: len(x[1]) > 1) \
        .persist()
    CS_list = CS_RDD.collect()                                 # [(c, [(i, [d, ...]), ...]), ...]

    CS_sum = CS_RDD.mapValues(summarize)\
        .collect()                                             # [(c, (N, SUM, SUMSQ)), ...]
    # print()
    # print('CS_sum')
    # print(CS_sum)
    RS_list = sepCSRSRDD.filter(lambda x: len(x[1]) == 1)\
        .map(lambda x: x[1][0])\
        .collect()                                             # [(i, [d,...]), ...]
    # [(c, [(i, [d, ...])]), ...]
    # print('RS_list', RS_list)

    # collect cluster result
    cluster_result_dict = {}
    for cp in DS_list:
        cluster_result_dict[cp[1][0]] = cp[0]
    for cps in CS_list:
        for ps in cps[1]:
            cluster_result_dict[ps[0]] = cps[0]
    for p in RS_list:
        cluster_result_dict[p[0]] = -1
    # print(cluster_result_dict)
    print()
    print('****************************************')
    print('in first chunk, there are', len(cluster_result_dict), 'points,')
    print(sum(x == -1 for x in cluster_result_dict.values()), 'points in RS')
    print('****************************************')
    print()
    # collect intermediate result
    round_result = intermediate(load_chunk_num, DS_sum, CS_sum, RS_list)
    output_b_list.append(round_result)

    first_time = time.time()                                                                                            # TIME!!!!!!!!!!
    print('no.%d load time:' % load_chunk_num, first_time - start_time)                                                 # TIME!!!!!!!!!!
    # print('results from first load:')
    # print('DS_sum')
    # print(DS_sum)
    # print('CS_sum')
    # print(CS_sum)
    # print('RS_list')
    # print(RS_list)


    # rest chunks
    for i in range(len(restPaths)):
        path = restPaths[i]
        load_chunk_num += 1
        this_start_time = time.time()
        # onePath = 'file:///Users/shengjiawu/Desktop/INF553/HW/HW5/hw5_data/test2/data1.txt'
        oneChunk = sc.textFile(path)\
            .map(readTxt)\
            .map(lambda p: assignSet(p, DS_sum, CS_sum))\
            .persist()
        # oneChunk (i1, ([d,...], [DS], [CS], [RS]))

        new_DS_sum = oneChunk.filter(lambda p_info: p_info[1][1])\
            .map(lambda p_info: (p_info[1][1][0], (p_info[0], p_info[1][0])))\
            .groupByKey()\
            .map(lambda x: (x[0], list(x[1])))\
            .map(lambda cp_l: updateDSorCS(cp_l, DS_sum))\
            .collect()
        # new_DS_c_list = list(set([c_info[0] for c_info in new_DS_sum]))
        # old_DS_c_list = list(set([c_info[0] for c_info in DS_sum]))
        # lost_DS_list = [c_info[0] for c_info in DS_sum if c_info not in new_DS_sum]
        # if lost_DS_list:
        #     for c in lost_DS_list:
        #         new_DS_sum.append((c, (0, 0, 0)))

        new_DS_sum = checkLostCluster(DS_sum, new_DS_sum)
        DS_sum = new_DS_sum

        new_CS_sum = oneChunk.filter(lambda p_info: p_info[1][2])\
            .map(lambda p_info: (p_info[1][2][0], (p_info[0], p_info[1][0])))\
            .groupByKey()\
            .map(lambda x: (x[0], list(x[1])))\
            .map(lambda cp_l: updateDSorCS(cp_l, CS_sum))\
            .collect()
        new_CS_sum = checkLostCluster(CS_sum, new_CS_sum)
        CS_sum = new_CS_sum
        RS_add_list = oneChunk.filter(lambda p_info: p_info[1][3])\
            .map(lambda p_info: (p_info[0], p_info[1][0]))\
            .collect()
        RS_list += RS_add_list
        len_RSbeforeKMeans = len(RS_list)
        # print(RS_list)

        print("RS_list length:", len_RSbeforeKMeans)
        if len_RSbeforeKMeans > FIRSTKTIMES * K:
            mixCSRS_list = KMeans(RS_list, FIRSTKTIMES * K, MAXITER) # [(c, (i, [d,...])),...]
            sepCSRSRDD = sc.parallelize(mixCSRS_list) \
                .groupByKey() \
                .map(lambda x: (x[0], list(x[1])))
            new_CS_RDD = sepCSRSRDD.filter(lambda x: len(x[1]) > 1) \
                .persist()
            RStoCS = new_CS_RDD.flatMap(lambda cps: [(p[0], cps[0]) for p in cps[1]])\
                .collect()                                             # [(i, c),...]

            new_CS_sum = new_CS_RDD.mapValues(summarize)\
                .collect()                                             # [(c, (N, SUM, SUMSQ)), ...]
            RS_list = sepCSRSRDD.filter(lambda x: len(x[1]) == 1)\
                .map(lambda x: x[1][0])\
                .collect()
            CS_sum, CStoCS = mergeCS(CS_sum, new_CS_sum, 'CSCS')
            print("RS_list length after kmeans:", len(RS_list))

        if load_chunk_num < total_file_num:
            # collect cluster result
            this_cluster_result_list = oneChunk.map(mapCluster).collect()  # [(i, c), ...]
            this_cluster_result_dict = dict(this_cluster_result_list)
            cluster_result_dict.update(this_cluster_result_dict)

            if len_RSbeforeKMeans > FIRSTKTIMES * K:
                new_cluster_result_dict = updateClusterResult_RS(cluster_result_dict, RStoCS)
                cluster_result_dict = new_cluster_result_dict
                new_cluster_result_dict = updateClusterResult_CS(cluster_result_dict, CStoCS)
                cluster_result_dict = new_cluster_result_dict

            print()
            print('****************************************')
            print('till no.%d chunk, there are in total' % load_chunk_num, len(cluster_result_dict), 'points,')
            print(sum(x == -1 for x in cluster_result_dict.values()), 'points in RS')
            print('****************************************')
            print()

            # collect intermediate result
            round_result = intermediate(load_chunk_num, DS_sum, CS_sum, RS_list)
            output_b_list.append(round_result)
            print(output_b_list)
            print('no.%d load time:' % load_chunk_num, time.time() - this_start_time)

    # refine last load
    DS_sum, CS_sum, CStoDS = mergeCS(DS_sum, CS_sum, 'DSCS')

    # collect cluster result
    this_cluster_result_list = oneChunk.map(mapCluster).collect()  # [(i, c), ...]
    this_cluster_result_dict = dict(this_cluster_result_list)
    cluster_result_dict.update(this_cluster_result_dict)

    if len_RSbeforeKMeans > FIRSTKTIMES * K:
        new_cluster_result_dict = updateClusterResult_RS(cluster_result_dict, RStoCS)
        cluster_result_dict = new_cluster_result_dict
        new_cluster_result_dict = updateClusterResult_CS(cluster_result_dict, CStoCS)
        cluster_result_dict = new_cluster_result_dict

    new_cluster_result_dict = updateClusterResult_CS(cluster_result_dict, CStoDS)
    cluster_result_dict = new_cluster_result_dict

    print()
    print('****************************************')
    print('till no.%d chunk, there are in total' % load_chunk_num, len(cluster_result_dict), 'points,')
    print(sum(x == -1 for x in cluster_result_dict.values()), 'points in RS')
    print('****************************************')
    print()

    # collect intermediate result
    round_result = intermediate(load_chunk_num, DS_sum, CS_sum, RS_list)
    output_b_list.append(round_result)

    print('no.%d load time:' % load_chunk_num, time.time() - this_start_time)
    # print(cluster_result_dict)

    # generate output files
    final_cluster_result_dict = {str(k): v for k, v in cluster_result_dict.items()}
    final_cluster_result_dict = OrderedDict(sorted(final_cluster_result_dict.items()))
    with open(output_file_cluster, 'w') as fo:
        json.dump(final_cluster_result_dict, fo)

    with open(output_file_intermediate, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerow(["round_id", "nof_cluster_discard", "nof_point_discard", "nof_cluster_compression", "nof_point_compression", "nof_point_retained"])
        writer.writerows(output_b_list)

    print('Duration:', time.time() - start_time)

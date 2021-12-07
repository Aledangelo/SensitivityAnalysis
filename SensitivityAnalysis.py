import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as pp
from sklearn.cluster import AgglomerativeClustering

workload = list()
# num_cluster = 20
# num_component = 4


def varTOT(data):
    t = 0
    for i in range(len(data)):
        t = pow(data.values[i] - data.values.mean(0), 2) + t

    # print(data.values.mean(0))
    tmp = 0
    for i in range(len(t)):
        tmp = tmp + t[i]

    return tmp


def varCL(intra, inter):
    tot = intra + inter
    intra_per = (intra * 100) / tot
    inter_per = 100 - intra_per

    return inter_per, intra_per


def createClusterDict(cluster):
    dict_cluster = dict()

    cl = sorted(cluster.labels_)

    for i in cl:
        dict_cluster[f'{i}'] = 0

    for i in cl:
        dict_cluster[f'{i}'] += 1

    return dict_cluster


def readData(name, path):
    data = pd.read_excel(str(path) + str(name) + '.xlsx')
    data = pd.DataFrame(data).drop(columns=["label", "responseCode", "responseMessage", "threadName", "dataType", "success", "URL", "IdleTime", "Principale1", "Principale2", "Principale3", "Cluster"])

    norm_file = stats.zscore(data)
    # norm_file = pd.DataFrame(norm_data).drop(columns=["label", "responseCode", "responseMessage", "threadName", "dataType", "success", "grpThreads", "allThreads", "URL", "Latency", "IdleTime", "Connect", "Principale1", "Principale2", "Principale3", "Cluster"])
    norm_path = str(path) + str(name) + '_normalized.csv'
    norm_file.to_csv(norm_path)

    return norm_file


def varPCA(data, n):
    pca = PCA(n_components=n)
    res = pca.fit_transform(data)

    v = 0
    for i in pca.explained_variance_ratio_:
        v += i
    v = v * 100
    varianceLostPCA = 100 - v

    return res, v, varianceLostPCA


name = input('Nome File: ')
num_component = int(input('Numero di Componenti Principali: '))
num_cluster = int(input('Numero di Cluster: '))
# name = 'PCA1_NO_OUTLIERS'
path = '/Users/alessandro/Documents/Università/Impianti di Elaborazione/esercizi/homework3_Pietro/'

norm_file = readData(name, path)

principalComponents, v, varLostPCA = varPCA(norm_file, num_component)
print('Componenti principali: ' + str(num_component))
print('Numero di Cluster: ' + str(num_cluster))

print('\nVarianza Mantenuta con PCA: ' + str(v) + ' %')
print('Varianza Persa con PCA: ' + str(varLostPCA) + ' %')

# pp.figure(figsize=(10, 7))
# pp.scatter(principalComponents[:, 0], principalComponents[:, 1])

try:
    linked = AgglomerativeClustering(n_clusters=num_cluster, linkage='ward', affinity='euclidean', compute_distances=True,
                                     compute_full_tree=False).fit(principalComponents)
except Exception as e:
    print(e)
    quit()

dict_cluster = createClusterDict(linked)
# print(dict_cluster)

W = 0
B = 0
for i in range(num_cluster):
    W_temp = 0
    B_temp = 0
    W_sum = 0
    B_sum = 0

    r = np.where(linked.labels_ == i)
    r = r[0]

    if len(r) == 1:
        centroid = principalComponents[r]
        centroid = centroid[0]
    else:
        centroid = principalComponents[r].mean(0)

    for j in range(len(principalComponents[r])):
        W_temp = pow(np.subtract(centroid, principalComponents[r][j]), 2) + W_temp

    for j in range(len(W_temp)):
        W_sum = W_temp[j] + W_sum

    for j in range(len(principalComponents.mean(0))):
        B_temp = pow(np.subtract(centroid, principalComponents.mean(0)), 2) + B_temp

    for j in range(len(B_temp)):
        B_sum = (B_temp[j]) + B_sum
    B_sum = B_sum * dict_cluster[f'{i}']

    W = W_sum + W
    B = B_sum + B

W_per, B_per = varCL(intra=W, inter=B)
print('\nVarianza Intra Cluster: ' + str(W_per) + ' %')
print('Varianza Inter Cluster: ' + str(B_per) + ' %')

sium = (varLostPCA + (W_per * v)) / 100
# sium = B / varTOT(norm_file)
print('\nVarianza Totale mantenuta: ' + str(sium) + ' %')
print('Varianza Totale Persa: ' + str(100 - sium) + ' %')

# print('\nDevianza totale: ' + str(B / varTOT(norm_file)) + ' %')
# print('\nB: ' + str(B))
# print('W: ' + str(W))
# print(B/varTOT(norm_file))

# print(norm_file.values[1])

# pp.figure(figsize=(10, 7))
# pp.scatter(linked[:, 0], linked[:, 1])
# pp.figure(figsize=(10, 7))
# dendrogram(linked, orientation='right')

#pp.show()

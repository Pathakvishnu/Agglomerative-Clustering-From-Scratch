from sklearn.metrics import accuracy_score,f1_score,classification_report
from sklearn.datasets import load_iris
from scipy import spatial
import numpy as np
import pandas as pd


class AgglomerativeClustering:
    
    def __init__(self,data,label,threshold=0.99):
        self.data = data
        self.label = label
        self.confidence = threshold
    
    def cosine_similarity(self,list_1, list_2,epsilon=1e-7):
        return np.dot(list_1, list_2) / ((np.linalg.norm(list_1) * np.linalg.norm(list_2))+epsilon)
    
    def agglomerative_clustering(self):
        sample_idx = [] # to store each sample as singleton cluster at the starting
        tot_sample = len(self.data)

        for i in range(tot_sample):
            sample_idx.append([i])

        i = 0
        while i<tot_sample:

            # below variable will reset after each iteration
            prev_len = len(sample_idx) 
            group = [] # store sample data which is similar 
            remove_idx = [] # index of sample data which is similar 

            # mean aggregation of the sample in same cluster
            group_avg_one = np.zeros(len(self.data[i]))
            for d in sample_idx[i]:
                group_avg_one+=self.data[d]

            group_avg_one = group_avg_one/len(sample_idx[i])

            j = i+1
            while j<tot_sample:

                # mean aggregation of the sample in same cluster
                group_avg_two = np.zeros(len(self.data[j]))
                for d in sample_idx[j]:
                    group_avg_two+=self.data[d]
                group_avg_two = group_avg_two/len(sample_idx[j])

                sim = self.cosine_similarity(group_avg_one,group_avg_two)
                if sim>=self.confidence:
                    group.extend(sample_idx[j])
                    remove_idx.append(j)

                j+=1

            # club into the cluster
            for sample_id in group:
                sample_idx[i].append(sample_id)

            # after clubbing we need to update the number of cluster in the data
            # so removing those index which has been clubbed in this iteration
            for idx,sample_id in enumerate(remove_idx):
                sample_idx.pop(sample_id-idx)

            # updaing the length of tot_sample to avoid index error
            tot_sample = len(sample_idx)
            i+=1

            # if there is no improvement in total number of cluster from previous step
            if len(sample_idx)==prev_len:
                break

        result = np.zeros(len(self.data))
        for i,sample_id in enumerate(sample_idx):
            for idx in sample_id:
                result[idx]=i
        return result
    
    def get_result(self,cluster):
        print(classification_report(binary_label,cluster))

data = load_iris()
sample_data = data['data']
binary_label = data['target']
label_name = data['target_names']

model = AgglomerativeClustering(sample_data,binary_label)
result = model.agglomerative_clustering()
model.get_result(result)
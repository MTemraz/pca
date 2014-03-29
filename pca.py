# Name: Mohamed Temraz
# Email: temraz11@gmail.com
# Description: Principal Components Analysis Implementation

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self,data,num_features=10):
        self.data = data
        self.features = num_features

    def pca(self):
        new_data = self.data - np.mean(self.data,axis=0)
        cov_data = np.cov(new_data)
        eig_vals,eig_vecs = np.linalg.eig(cov_data)
        ind = list(reversed(eig_vals.argsort()))[:self.features]
        top_eig_vecs = eig_vecs[:,ind]
        red_data = np.mat(new_data) * np.mat(top_eig_vecs)
        recon = (red_data*np.mat(top_eig_vecs.T)) + np.mean(self.data,axis=0)
        return recon

    def plot(self,recon):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.data[:,0],self.data[:,1],marker='o',c='blue')
        ax.scatter(recon[:,0],recon[:,1],marker='^',c='red')



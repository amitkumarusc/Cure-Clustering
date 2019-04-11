#!/usr/bin/python
# coding: utf-8

import sys
import heapq as hq
import numpy as np
import itertools
from random import shuffle
from matplotlib import pyplot as plt

def readDataset():
	data = {'x_data': [], 'y_data': []}
	with open('iris.dat') as file_handle:
		lines = file_handle.readlines()
		for line in lines:
			point = line.strip().split(',')
			label = point[-1]
			point = point[:-1]
			point = map(float, point)
			data['x_data'].append(point)
			data['y_data'].append(label)
	return data


def plotGoldStandards(i, j):
	points = np.array(data['x_data'])
	labels = data['y_data']
	mapping = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}
	colors = [mapping[label] for label in labels]
	f, ax = plt.subplots(1)
	plt.scatter(points[:, i], points[:,j], c=colors)
	ax.legend()
	plt.show()

class Cluster(object):
	COUNTER = 0
	def __init__(self, points, p_ids, rep_count, alpha):
		Cluster.COUNTER += 1
		self.id = Cluster.COUNTER
		self.points = np.array(points) if type(points) == type([]) else points
		self.n_points, self.dimensions = self.points.shape
		self.p_ids = p_ids
		self.mean = None
		self.alpha = alpha
		self.rep_count = rep_count
		self.reps = self.assignReps()
		self.closest = None
		self.closestDist = sys.maxint

	def getMean(self):
		if self.mean is not None:
			return self.mean
		self.mean = self.points.sum(axis=0)/(self.n_points*1.0)
		return self.mean
	
	def assignReps(self):
		if self.n_points <= self.rep_count:
			return self.points[:,:]
		tmp_set = set()
		reps = []
		for i in range(self.rep_count):
			max_dist = 0
			for j in range(self.n_points):
				if i == 0:
					min_dist = Cluster.p_distance(self.points[j], self.getMean())
				else:
					min_dist = self.getClosestDist(self.points[j], reps)
				if min_dist >= max_dist:
					max_dist = min_dist
					max_point = j
			if max_point not in tmp_set:
				tmp_set.add(max_point)
				if reps is not None:
					point = self.points[max_point]
					reps.append(point + self.alpha * (self.getMean() - point))
				else:
					point = self.points[max_point]
					reps = [point + self.alpha * (self.getMean() - point)]
		reps = np.array(reps)
		return reps
			
					
	def getClosestDist(self, point, points):
		points = np.array(points)
		return min(np.sqrt(np.sum((points - point)**2, axis=1)))
	
	def __str__(self):
		return "[ID:%d] PointCount=%d, ClosestDist=%0.2f with [ID:%d]"%(self.id, self.n_points, self.closestDist, self.closest.id if self.closest else -1)
	
	def __repr(self):
		return self.__str__()

	@classmethod
	def merge(cls, cluster1, cluster2):
		if cluster1.dimensions != cluster2.dimensions:
			raise ValueError('Error! The dimensions of the data-points does not match.')
		
		combined_points = np.concatenate((cluster1.points, cluster2.points))
		combined_p_ids = cluster1.p_ids + cluster2.p_ids

		new_cluster = Cluster(combined_points, combined_p_ids, cluster1.rep_count, cluster1.alpha)
		
		new_cluster.mean = (cluster1.getMean()*cluster1.n_points + cluster2.getMean()*cluster2.n_points)/(new_cluster.n_points*1.0)

		return new_cluster
	
	@staticmethod
	def p_distance(point1, point2):
		return np.sqrt(np.sum((point1 - point2)**2))

	@staticmethod
	def c_distance(cluster1, cluster2):
		min_dist = sys.maxint
		for rep_1_index in xrange(len(cluster1.reps)):
			for rep_2_index in xrange(len(cluster2.reps)):
				dist = Cluster.p_distance(cluster1.reps[rep_1_index], cluster2.reps[rep_2_index])
				min_dist = min(min_dist, dist)
		return min_dist

class HeirarchicalClustering(object):
	def __init__(self, cluster_count, rep_count, alpha):
		self.cluster_count = cluster_count
		self.clusters = []
		self.rep_count = rep_count
		self.alpha = alpha

	def fit(self, points):
		for p_index, point in enumerate(points):
			self.clusters.append(Cluster([point], [p_index], self.rep_count, self.alpha))
		
		for index in xrange(len(self.clusters)):
			self.assignClosestCluster(self.clusters[index])
			
		while len(self.clusters) > self.cluster_count:
			u_index = self.getMergeCandidate()
			u_cluster = self.clusters.pop(u_index)
			v_cluster = u_cluster.closest
			
			self.removeCluster(v_cluster)
			
			new_cluster = Cluster.merge(u_cluster, v_cluster)
			
			self.rebuildClosestClusterLinks(new_cluster, u_cluster.id, v_cluster.id)
			
			self.clusters.append(new_cluster)

	def predict(self):
		pass
	
	def getMergeCandidate(self):
		min_dist = sys.maxint
		cluster_index = 0
		for i in range(len(self.clusters)):
			dist = self.clusters[i].closestDist
			if dist < min_dist:
				min_dist = dist
				cluster_index = i
		return cluster_index
			
	def assignClosestCluster(self, cluster):
		min_dist = sys.maxint
		closest = None
		for i in range(len(self.clusters)):
			if cluster.id == self.clusters[i].id: continue
			
			dist = Cluster.c_distance(cluster, self.clusters[i])
			if dist < min_dist:
				min_dist = dist
				closest = self.clusters[i]
		cluster.closest = closest
		cluster.closestDist = min_dist

	def rebuildClosestClusterLinks(self, new_cluster, cid_1, cid_2):
		min_dist = sys.maxint
		for i in range(len(self.clusters)):
			new_dist = Cluster.c_distance(new_cluster, self.clusters[i])
			if new_cluster.closestDist > new_dist:
				new_cluster.closestDist = new_dist
				new_cluster.closest = self.clusters[i]

			if self.clusters[i].closest.id in (cid_1, cid_2):
				if self.clusters[i].closestDist < new_dist:
					self.assignClosestCluster(self.clusters[i])
				else:
					self.clusters[i].closestDist = new_dist
					self.clusters[i].closest = new_cluster
			else:
				if self.clusters[i].closestDist > new_dist:
					self.clusters[i].closestDist = new_dist
					self.clusters[i].closest = new_cluster
	
	def removeCluster(self, cluster):
		for i in range(len(self.clusters)):
			if self.clusters[i].id == cluster.id:
				self.clusters[i] = self.clusters[-1]
				self.clusters.pop()
				return
		print "Cluster not found : ", cluster.id, cluster.p_ids


def gridSearch():
	rep_counts = list(range(1, 6))
	alphas = np.arange(0.1, 1, 0.1)
	for rep_count in rep_counts:
		for alpha in alphas:
			hc = HeirarchicalClustering(3, rep_count, alpha)
			hc.fit(data['x_data'])
			plotPoints(hc.clusters)
			print 'RepCount: %d, Alpha: %f'%(rep_count, alpha)
			


def plotPoints(clusters):
	f, ax = plt.subplots(1)
	for i, cluster in enumerate(clusters):
		points = np.array(cluster.points)
		plt.scatter(points[:, 0], points[:, 2], label=i)
	ax.legend()
	plt.show()


def calculatMetrics(pred_clusters, true_labels_arr):
	true_labels = {}
	for index, true_label in enumerate(true_labels_arr):
		if true_label in true_labels:
			true_labels[true_label].append(index)
		else:
			true_labels[true_label] = [index]
	true_labels = true_labels.values()
	true_label_pairs = []
	for true_cluster in true_labels:
		true_cluster.sort()
		true_label_pairs += itertools.combinations(true_cluster, 2)
	
	pred_label_pairs = []
	for pred_cluster in pred_clusters:
		p_ids = pred_cluster.p_ids
		p_ids.sort()
		pred_label_pairs += itertools.combinations(p_ids, 2)
	print len(true_label_pairs), len(pred_label_pairs)
	true_label_pairs = set(true_label_pairs)
	pred_label_pairs = set(pred_label_pairs)
	print len(true_label_pairs), len(pred_label_pairs)
	
	precision = len(pred_label_pairs.intersection(true_label_pairs))/(1.0*len(pred_label_pairs))
	recall = len(pred_label_pairs.intersection(true_label_pairs))/(1.0*len(true_label_pairs))
	print "Precision: %0.2f, Recall: %0.2f"%(precision, recall)


if __name__ == '__main__':
	data = readDataset()
	cluster_count = 3
	rep_count = 2
	alpha = 0.8

	hc = HeirarchicalClustering(3, 2, 0.5)
	hc.fit(data['x_data'])
	plotPoints(hc.clusters)

	gridSearch()

	plotGoldStandards(0, 2)

	calculatMetrics(hc.clusters, data['y_data'])



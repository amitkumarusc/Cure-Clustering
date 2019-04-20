"""
Name: Amit Kumar
USC ID: 7088752227
Email: kuma310@usc.edu
"""

import sys
import numpy as np
import itertools

def readDataset(data_file):
	data = {'x_data': [], 'y_data': []}
	with open(data_file) as file_handle:
		lines = file_handle.readlines()
		for line in lines:
			point = line.strip().split(',')
			label = point[-1]
			point = point[:-1]
			point = map(float, point)
			data['x_data'].append(point)
			data['y_data'].append(label)
	return data

def readSample(sample_file):
	indexes = []
	with open(sample_file) as file_handle:
		lines = file_handle.readlines()
		for line in lines:
			indexes.append(int(line.strip()))
	return indexes

def getSamplePoints(sample_indexes, data):
	sample_data = {'x_data': [], 'y_data': []}
	for index in sample_indexes:
		sample_data['x_data'].append(data['x_data'][index])
		sample_data['y_data'].append(data['y_data'][index])
	return sample_data

def calculateMetrics(pred_clusters, true_labels_arr):
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
	for index, pred_cluster in enumerate(pred_clusters):
		pred_cluster.sort()
		print 'Cluster %d: %s'%(index+1, str(pred_cluster))
		pred_label_pairs += itertools.combinations(pred_cluster, 2)
	true_label_pairs = set(true_label_pairs)
	pred_label_pairs = set(pred_label_pairs)
	
	precision = len(pred_label_pairs.intersection(true_label_pairs))/(1.0*len(pred_label_pairs))
	recall = len(pred_label_pairs.intersection(true_label_pairs))/(1.0*len(true_label_pairs))
	f1 = 2*precision*recall/(precision+recall)
	print "Precision: %f, Recall: %f"%(precision, recall)
	# print "F1 score: %f"%(f1)
	print '\n'

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
	
	def pointToClusterDistance(self, point):
		min_dist = sys.maxint
		for rep_index in xrange(len(self.reps)):
			dist = Cluster.p_distance(self.reps[rep_index], point)
			min_dist = min(min_dist, dist)
		return min_dist
	
	@staticmethod
	def p_distance(point1, point2):
		return np.sqrt(np.sum((point1 - point2)**2))

	@staticmethod
	def c_distance(cluster1, cluster2):
		# Comment the line below if the distance between clusters is distance between their closest representatives
		# There were contradicting statements in discussion forum and announcements.
		return Cluster.p_distance(cluster1.getMean(), cluster2.getMean())

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
	
	def predict(self, points):
		points = np.array(points)
		predictions = {cluster.id:[] for cluster in self.clusters}
		for i in range(len(points)):
			label = self.predict_point(points[i])
			predictions[label].append(i)
		return predictions.values()

	def predict_point(self, point):
		point = np.array(point)
		min_dist = sys.maxint
		label = None
		for i in range(len(self.clusters)):
			dist = self.clusters[i].pointToClusterDistance(point)
			if dist < min_dist:
				min_dist = dist
				label = self.clusters[i].id
		return label
	
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
		print("Cluster not found : ", cluster.id, cluster.p_ids)

if __name__ == '__main__':
	
	if len(sys.argv) < 6:
		print 'Usage: python cure.py <k> <sample> <data-set> <n> <alpha>'
		sys.exit(-1)

	cluster_count = int(sys.argv[1])
	sample_file = sys.argv[2]
	data_file = sys.argv[3]
	rep_count = int(sys.argv[4])
	alpha = float(sys.argv[5])

	data = readDataset(data_file)
	sample_indexes = readSample(sample_file)

	sample = getSamplePoints(sample_indexes, data)

	hc = HeirarchicalClustering(cluster_count, rep_count, alpha)
	hc.fit(sample['x_data'])
	predictions = hc.predict(data['x_data'])
	calculateMetrics(predictions, data['y_data'])


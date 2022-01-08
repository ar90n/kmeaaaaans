package kmeaaaaans

import (
	"gonum.org/v1/gonum/mat"
)

type miniBatchKmeans struct {
	nClusters     uint
	batchSize     uint
	tolerance     float64
	maxIterations uint
}

var _ Kmeans = (*miniBatchKmeans)(nil)

func (k *miniBatchKmeans) Fit(X *mat.Dense) TrainedKmeans {
	centroids := calcInitialCentroids(X, k.nClusters, KmeansPlusPlus)
	return &trainedKmeans{
		centroids: centroids,
	}
}

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

func (k *miniBatchKmeans) Fit(X *mat.Dense) TrainedKmeans {
	centroids := calcInitialCentroids(X, k.nClusters)
	return &trainedKmeans{
		centroids: centroids,
	}
}

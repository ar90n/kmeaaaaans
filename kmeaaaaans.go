package kmeaaaaans

import (
	"gonum.org/v1/gonum/mat"
)

type Kmeans interface {
	Fit(X *mat.Dense) TrainedKmeans
}

type TrainedKmeans interface {
	Predict(X *mat.Dense) []uint
	Centroids() *mat.Dense
}

func NewMiniBatchKmeans(nClusters uint, batchSize uint) Kmeans {
	return &miniBatchKmeans{
		nClusters: nClusters,
		batchSize: batchSize,
	}
}

func NewVanilaKmeans(nClusters uint, tolerance float64, maxIterations uint, chunkSize uint) Kmeans {
	return &vanilaKmeans{
		nClusters:     nClusters,
		tolerance:     tolerance,
		maxIterations: maxIterations,
		chunkSize:     chunkSize,
	}
}

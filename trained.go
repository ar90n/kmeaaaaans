package kmeaaaaans

import (
	"gonum.org/v1/gonum/mat"
)

type trainedKmeans struct {
	centroids *mat.Dense
}

func (k *trainedKmeans) Predict(X *mat.Dense) []uint {
	indices := makeSequence(uint(X.RawMatrix().Rows))
	classes := make([]uint, X.RawMatrix().Rows)
	assignCluster(X, k.centroids, classes, indices, calcL2Distance)
	return classes
}

func (k *trainedKmeans) Centroids() *mat.Dense {
	return mat.DenseCopyOf(k.centroids)
}

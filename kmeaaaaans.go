package kmeaaaaans

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type UpdateAlgorithm int

const (
	Lloyd UpdateAlgorithm = iota + 1
	MiniBatch
)

func UpdateAlgorithmFrom(str string) (UpdateAlgorithm, error) {
	switch str {
	case "lloyd":
		return Lloyd, nil
	case "mini-batch":
		return MiniBatch, nil
	default:
		return 0, fmt.Errorf("invalid update algorithm: %s", str)
	}
}

type Kmeans interface {
	Fit(X *mat.Dense) (TrainedKmeans, error)
}

type TrainedKmeans interface {
	Predict(X *mat.Dense) []uint
	Centroids() *mat.Dense
}

func NewMiniBatchKmeans(nClusters uint, tolerance float64, maxIterations uint, batchSize uint, initAlgorithm InitAlgorithm) Kmeans {
	return &miniBatchKmeans{
		tolerance:     tolerance,
		maxIterations: maxIterations,
		nClusters:     nClusters,
		batchSize:     batchSize,
		initAlgorithm: initAlgorithm,
	}
}

func NewVanilaKmeans(nClusters uint, tolerance float64, maxIterations uint, chunkSize uint, initAlgorithm InitAlgorithm) Kmeans {
	return &vanilaKmeans{
		nClusters:     nClusters,
		tolerance:     tolerance,
		maxIterations: maxIterations,
		chunkSize:     chunkSize,
		initAlgorithm: initAlgorithm,
	}
}

func NewTrainedKmeans(centroids *mat.Dense) TrainedKmeans {
	return &trainedKmeans{
		centroids: centroids,
	}
}

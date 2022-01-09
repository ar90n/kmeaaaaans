package kmeaaaaans

import (
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type miniBatchKmeans struct {
	nClusters     uint
	tolerance     float64
	maxIterations uint
	batchSize     uint
	initAlgorithm InitAlgorithm
}

var _ Kmeans = (*miniBatchKmeans)(nil)

func makeBatchIndice(nSamples, batchSize uint) []uint {
	batchIndice := make([]uint, batchSize)
	for i := uint(0); i < batchSize; i++ {
		batchIndice[i] = uint(rand.Intn(int(nSamples)))
	}
	return batchIndice
}

func (k *miniBatchKmeans) Fit(X *mat.Dense) TrainedKmeans {
	nextCentroids := calcInitialCentroids(X, k.nClusters, k.initAlgorithm)
	centroids := mat.NewDense(nextCentroids.RawMatrix().Rows, nextCentroids.RawMatrix().Cols, nil)

	indices := makeBatchIndice(uint(X.RawMatrix().Rows), k.batchSize)
	chunks := makeChunks(indices, 1024)
	classes := make([]uint, X.RawMatrix().Rows)
	for i := 0; i < int(k.maxIterations) && k.tolerance < calcError(centroids, nextCentroids); i++ {
		centroids, nextCentroids = nextCentroids, centroids
		nextCentroids.Zero()

		nSamplesInCluster := make([]uint, k.nClusters)
		var wg sync.WaitGroup
		for _, chunk := range chunks {
			wg.Add(1)
			go func(chunk []uint) {
				defer wg.Done()
				assignCluster(X, centroids, classes, chunk, calcL2Distance)
				accumulateSamplesInCluster(X, nextCentroids, nSamplesInCluster, classes, chunk)
			}(chunk)
		}
		wg.Wait()
		updateCentroid(X, nextCentroids, nSamplesInCluster)
	}
	centroids = nextCentroids

	return &trainedKmeans{
		centroids: centroids,
	}

}

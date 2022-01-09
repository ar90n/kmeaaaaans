package kmeaaaaans

import (
	"runtime"
	"sync"

	"github.com/panjf2000/ants/v2"
	"gonum.org/v1/gonum/mat"
)

type vanilaKmeans struct {
	nClusters     uint
	tolerance     float64
	maxIterations uint
	chunkSize     uint
	initAlgorithm InitAlgorithm
}

var _ Kmeans = (*vanilaKmeans)(nil)

func accumulateSamples(X *mat.Dense, nextCentroids *mat.Dense, nSamplesInCluster []uint, classes []uint, indices []uint) {
	for i := 0; i < len(nSamplesInCluster); i++ {
		nSamplesInCluster[i] = 0
	}

	for _, i := range indices {
		cluster := int(classes[i])
		nSamplesInCluster[cluster]++
		centroidData := nextCentroids.RawRowView(cluster)
		featData := X.RawRowView(int(i))
		for j := 0; j < len(featData); j++ {
			centroidData[j] += featData[j]
		}
	}
}

func updateCentroids(centroids, nextCentroids *mat.Dense, nSamplesInCluster []uint) {
	for i := 0; i < len(nSamplesInCluster); i++ {
		if 0 < nSamplesInCluster[i] {
			scale := 1.0 / float64(nSamplesInCluster[i])
			centroidData := nextCentroids.RawRowView(i)
			for j := 0; j < nextCentroids.RawMatrix().Cols; j++ {
				centroidData[j] *= scale
			}
		} else {
			nextCentroids.SetRow(i, centroids.RawRowView(i))
		}
	}
}

func (k *vanilaKmeans) Fit(X *mat.Dense) TrainedKmeans {
	nSamples, featDim := X.Dims()
	nextCentroids := calcInitialCentroids(X, k.nClusters, k.initAlgorithm)
	centroids := mat.NewDense(int(k.nClusters), int(featDim), nil)

	indices := makeSequence(uint(nSamples))
	chunks := makeChunks(indices, k.chunkSize)
	classes := make([]uint, nSamples)

	type AssignClusterHandlerParams struct {
		X         *mat.Dense
		Centroids *mat.Dense
		Classes   []uint
		Indices   []uint
	}

	var wg sync.WaitGroup
	defer ants.Release()
	assignClusterHandler, _ := ants.NewPoolWithFunc(runtime.NumCPU(), func(i interface{}) {
		defer wg.Done()
		param := i.(AssignClusterHandlerParams)
		assignCluster(param.X, param.Centroids, param.Classes, param.Indices, calcL2Distance)
	})
	defer assignClusterHandler.Release()

	nSamplesInCluster := make([]uint, k.nClusters)
	for i := 0; i < int(k.maxIterations) && k.tolerance < calcError(centroids, nextCentroids); i++ {
		centroids, nextCentroids = nextCentroids, centroids
		nextCentroids.Zero()

		for _, chunk := range chunks {
			wg.Add(1)
			params := AssignClusterHandlerParams{
				X:         X,
				Centroids: centroids,
				Classes:   classes,
				Indices:   chunk,
			}
			assignClusterHandler.Invoke(params)
		}
		wg.Wait()

		accumulateSamples(X, nextCentroids, nSamplesInCluster, classes, indices)
		updateCentroids(centroids, nextCentroids, nSamplesInCluster)
	}
	centroids = nextCentroids

	return &trainedKmeans{
		centroids: centroids,
	}
}

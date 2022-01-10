package kmeaaaaans

import (
	"runtime"
	"sync"

	"github.com/panjf2000/ants/v2"
	"gonum.org/v1/gonum/mat"
)

type lloydKmeans struct {
	nClusters     uint
	tolerance     float64
	maxIterations uint
	chunkSize     uint
	initAlgorithm InitAlgorithm
}

var _ Kmeans = (*lloydKmeans)(nil)

func updateLloydCentroids(centroids, nextCentroids *mat.Dense, nSamplesInCluster []uint) {
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

func (k *lloydKmeans) Fit(X *mat.Dense) (TrainedKmeans, error) {
	nSamples, featDim := X.Dims()
	nextCentroids := calcInitialCentroids(X, k.nClusters, k.initAlgorithm)
	centroids := mat.NewDense(int(k.nClusters), int(featDim), nil)

	defer ants.Release()
	pool, err := ants.NewPool(runtime.NumCPU())
	if err != nil {
		return &trainedKmeans{}, nil
	}
	defer pool.Release()

	classes := make([]uint, nSamples)
	indices := makeSequence(uint(nSamples))
	chunks := makeChunks(indices, k.chunkSize)
	nSamplesInCluster := make([]uint, k.nClusters)
	for i := 0; i < int(k.maxIterations) && k.tolerance < calcError(centroids, nextCentroids); i++ {
		centroids, nextCentroids = nextCentroids, centroids

		var wg sync.WaitGroup
		for _, chunk := range chunks {
			chunk := chunk
			wg.Add(1)
			pool.Submit(func() {
				defer wg.Done()
				assignCluster(X, centroids, classes, chunk, calcL2Distance)
			})
		}
		wg.Wait()

		accumulateSamples(X, nextCentroids, nSamplesInCluster, classes, indices)
		updateLloydCentroids(centroids, nextCentroids, nSamplesInCluster)
	}
	centroids = nextCentroids

	return &trainedKmeans{
		centroids: centroids,
	}, nil
}

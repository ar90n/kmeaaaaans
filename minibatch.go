package kmeaaaaans

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/panjf2000/ants/v2"
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

func updateMiniBatchCentroids(nextCentroids *mat.Dense, centroids *mat.Dense, nSamplesInCluster []uint, accNSamplesInCluster []uint) {
	for i := 0; i < len(nSamplesInCluster); i++ {
		accNSamplesInCluster[i] += nSamplesInCluster[i]
		if 0 < nSamplesInCluster[i] {
			w0 := 1.0 / float64(accNSamplesInCluster[i])
			nextCentroidRowData := nextCentroids.RawRowView(i)
			w1 := w0 * float64(nSamplesInCluster[i])
			curCentroidRowData := centroids.RawRowView(i)
			for j := 0; j < nextCentroids.RawMatrix().Cols; j++ {
				nextCentroidRowData[j] = w0*nextCentroidRowData[j] + (1-w1)*curCentroidRowData[j]
			}
		} else {
			nextCentroids.SetRow(i, centroids.RawRowView(i))
		}
	}
}

func (k *miniBatchKmeans) Fit(X *mat.Dense) (TrainedKmeans, error) {
	nSamples, featDim := X.Dims()
	nextCentroids := calcInitialCentroids(X, k.nClusters, k.initAlgorithm)
	centroids := mat.NewDense(int(k.nClusters), featDim, nil)

	defer ants.Release()
	pool, err := ants.NewPool(runtime.NumCPU())
	if err != nil {
		return &trainedKmeans{}, nil
	}
	defer pool.Release()

	classes := make([]uint, X.RawMatrix().Rows)
	accNSamplesInCluster := make([]uint, k.nClusters)
	nSamplesInCluster := make([]uint, k.nClusters)
	minInertia := math.MaxFloat64
	minRuns := 0
	for i := 0; i < int(k.maxIterations) && k.tolerance < calcError(centroids, nextCentroids); i++ {
		centroids, nextCentroids = nextCentroids, centroids
		nextCentroids.Zero()

		indices := makeBatchIndice(uint(nSamples), k.batchSize)
		chunks := makeChunks(indices, 256)

		inertia := 0.0
		var wg sync.WaitGroup
		var mu sync.Mutex
		for _, chunk := range chunks {
			chunk := chunk
			wg.Add(1)
			pool.Submit(func() {
				defer wg.Done()
				partialInertia := assignCluster(X, centroids, classes, chunk, calcL2Distance)

				mu.Lock()
				defer mu.Unlock()
				inertia += partialInertia
			})
		}
		wg.Wait()

		if inertia < minInertia {
			minInertia = inertia
			minRuns = 0
		} else {
			minRuns++
		}
		if minRuns > 10 {
			break
		}

		accumulateSamples(X, nextCentroids, nSamplesInCluster, classes, indices)
		updateMiniBatchCentroids(nextCentroids, centroids, nSamplesInCluster, accNSamplesInCluster)
	}
	centroids = nextCentroids

	return &trainedKmeans{
		centroids: centroids,
	}, nil
}

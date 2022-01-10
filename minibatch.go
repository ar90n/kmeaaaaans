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
	maxNoImprobe  uint
	batchSize     uint
	initAlgorithm InitAlgorithm
}

var _ Kmeans = (*miniBatchKmeans)(nil)

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
	chunkSize := (k.batchSize + uint(runtime.NumCPU()) - 1) / uint(runtime.NumCPU())
	minInertia := math.MaxFloat64
	minRuns := uint(0)
	allIndices := makeSequence(uint(nSamples))
	getBegAndEnd := func(i int, nIndex int, batchSize uint) (int, int) {
		end := (i + 1) * int(batchSize)
		if nIndex < end {
			end = minInt(nIndex, int(batchSize))
		}
		beg := maxInt(0, end-int(batchSize))
		return beg, end
	}
	for i := 0; i < int(k.maxIterations) && k.tolerance < calcError(centroids, nextCentroids); i++ {
		centroids, nextCentroids = nextCentroids, centroids
		nextCentroids.Zero()

		beg, end := getBegAndEnd(i, len(allIndices), k.batchSize)
		if beg == 0 {
			rand.Shuffle(len(allIndices), func(i, j int) { allIndices[i], allIndices[j] = allIndices[j], allIndices[i] })
		}
		indices := allIndices[beg:end]
		chunks := makeChunks(indices, chunkSize)

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
		if k.maxNoImprobe < minRuns {
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

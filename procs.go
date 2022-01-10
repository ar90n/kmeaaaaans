package kmeaaaaans

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type InitAlgorithm int

const (
	KmeansPlusPlus InitAlgorithm = iota + 1
	Random
)

func InitAlgorithmFrom(str string) (InitAlgorithm, error) {
	switch str {
	case "kmeans++":
		return KmeansPlusPlus, nil
	case "random":
		return Random, nil
	default:
		return 0, fmt.Errorf("invalid init algorithm: %s", str)
	}
}

func calcRandomInitialCentroids(X *mat.Dense, nClusters uint) *mat.Dense {
	_, nFeatures := X.Dims()
	centroids := mat.NewDense(int(nClusters), nFeatures, nil)
	for i := 0; i < int(nClusters); i++ {
		for j := 0; j < int(nFeatures); j++ {
			centroids.Set(i, j, math.Abs(rand.NormFloat64()))
		}
	}
	return centroids
}

func calcKmeansPlusPlusInitialCentroids(X *mat.Dense, nClusters uint) *mat.Dense {
	nSamples, featDim := X.Dims()
	centroids := mat.NewDense(int(nClusters), featDim, nil)
	centroids.SetRow(0, X.RawRowView(rand.Intn(int(nSamples))))
	for i := 1; i < int(nClusters); i++ {
		accDistances := make([]float64, nSamples)
		for j := 0; j < int(nSamples); j++ {
			minDinstance := math.MaxFloat64
			for k := 0; k < i; k++ {
				distance := calcL2Distance(X.RawRowView(j), centroids.RawRowView(k))
				minDinstance = math.Min(minDinstance, distance)
			}
			if j == 0 {
				accDistances[j] = minDinstance
			} else {
				accDistances[j] = accDistances[j-1] + minDinstance
			}
		}

		r := accDistances[nSamples-1] * rand.Float64()
		j := sort.Search(int(nSamples), func(i int) bool { return accDistances[i] >= r })
		centroids.SetRow(i, X.RawRowView(j))
	}

	return centroids
}

func calcInitialCentroids(X *mat.Dense, nClusters uint, initAlgorithm InitAlgorithm) *mat.Dense {
	switch initAlgorithm {
	case KmeansPlusPlus:
		return calcKmeansPlusPlusInitialCentroids(X, nClusters)
	case Random:
		return calcRandomInitialCentroids(X, nClusters)
	default:
		panic("invalid init algorithm")
	}
}

func assignCluster(X *mat.Dense, centroids *mat.Dense, classes []uint, indices []uint, calcDistance func(X, Y []float64) float64) float64 {
	nClusters, _ := centroids.Dims()
	inertia := 0.0
	for _, i := range indices {
		minDist := math.MaxFloat64
		minClass := uint(0)
		for j := 0; j < int(nClusters); j++ {
			dist := calcDistance(X.RawRowView(int(i)), centroids.RawRowView(j))
			if dist < minDist {
				minDist = dist
				minClass = uint(j)
			}
		}
		inertia += minDist
		classes[i] = minClass
	}

	return inertia
}

func accumulateSamples(X *mat.Dense, nextCentroids *mat.Dense, nSamplesInCluster []uint, classes []uint, indices []uint) {
	nextCentroids.Zero()
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

func calcL2Distance(X, Y []float64) float64 {
	acc := 0.0
	i := 0
	for ; i < len(X)%4; i++ {
		diff := X[i] - Y[i]
		acc += diff * diff
	}

	for ; i < len(X); i += 4 {
		diff0 := X[i] - Y[i]
		diff1 := X[i+1] - Y[i+1]
		diff2 := X[i+2] - Y[i+2]
		diff3 := X[i+3] - Y[i+3]
		acc += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
	}

	return math.Sqrt(acc)
}

func calcError(X, Y *mat.Dense) float64 {
	return calcL2Distance(X.RawMatrix().Data, Y.RawMatrix().Data) / mat.Norm(X, 2)
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func makeSequence(max uint) []uint {
	seq := make([]uint, max)
	for i := range seq {
		seq[i] = uint(i)
	}
	return seq
}

func minUint(a, b uint) uint {
	if a < b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxUint(a, b uint) uint {
	if a < b {
		return b
	}
	return a
}

func maxInt(a, b int) int {
	if a < b {
		return b
	}
	return a
}

func makeChunks(seq []uint, chunkSize uint) [][]uint {
	chunks := make([][]uint, 0)
	for i := uint(0); i < uint(len(seq)); i += chunkSize {
		beg := i
		end := minUint(i+chunkSize, uint(len(seq)))
		chunks = append(chunks, seq[beg:end])
	}
	return chunks
}

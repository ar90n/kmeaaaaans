package kmeaaaaans

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type Kmeans interface {
	Fit(X *mat.Dense) TrainedKmeans
}

type TrainedKmeans interface {
	Predict(X *mat.Dense) []uint
	Centroids() *mat.Dense
}

type trainedKmeans struct {
	centroids *mat.Dense
}

func (k *trainedKmeans) Predict(X *mat.Dense) []uint {
	classes := make([]uint, X.RawMatrix().Rows)
	l2_dist := func(X, Y mat.Vector) float64 {
		diff := mat.NewVecDense(X.Len(), nil)
		diff.SubVec(X, Y)
		return mat.Norm(diff, 2)
	}
	assignCluster(X, k.centroids, classes, l2_dist)
	return classes
}

func (k *trainedKmeans) Centroids() *mat.Dense {
	return mat.DenseCopyOf(k.centroids)
}

type vanilaKmeans struct {
	nClusters     uint
	tolerance     float64
	maxIterations uint
}

func calcInitialCentroids(X *mat.Dense, nClusters uint) *mat.Dense {
	_, nFeatures := X.Dims()
	centroids := mat.NewDense(int(nClusters), nFeatures, nil)
	for i := 0; i < int(nClusters); i++ {
		for j := 0; j < int(nFeatures); j++ {
			centroids.Set(i, j, rand.NormFloat64())
		}
	}
	return centroids
}

func assignCluster(X *mat.Dense, centroids *mat.Dense, classes []uint, calcDistance func(X, Y mat.Vector) float64) {
	nClusters, _ := centroids.Dims()
	nSamples, _ := X.Dims()
	for i := 0; i < nSamples; i++ {
		minDist := math.MaxFloat64
		minClass := uint(0)
		for j := 0; j < int(nClusters); j++ {
			dist := calcDistance(X.RowView(i), centroids.RowView(j))
			if dist < minDist {
				minDist = dist
				minClass = uint(j)
			}
		}
		classes[i] = minClass
	}
}

func updateCentroid(X *mat.Dense, centroids *mat.Dense, classes []uint) {
	nClusters, featDim := centroids.Dims()
	nSamples, _ := X.Dims()

	nSamplesInCluster := make([]int, nClusters)
	centroids.Zero()
	for i := 0; i < nSamples; i++ {
		feat := X.RowView(i)
		cluster := int(classes[i])
		for j := 0; j < feat.Len(); j++ {
			accValue := centroids.At(cluster, j) + feat.AtVec(j)
			centroids.Set(cluster, j, accValue)
		}
		nSamplesInCluster[cluster]++
	}

	for i := 0; i < int(nClusters); i++ {
		for j := 0; j < featDim; j++ {
			avgValue := centroids.At(i, j) / float64(nSamplesInCluster[i])
			centroids.Set(i, j, avgValue)
		}
	}
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func (k *vanilaKmeans) Fit(X *mat.Dense) TrainedKmeans {
	centroids := calcInitialCentroids(X, k.nClusters)

	classes := make([]uint, X.RawMatrix().Rows)
	l2_dist := func(X, Y mat.Vector) float64 {
		diff := mat.NewVecDense(X.Len(), nil)
		diff.SubVec(X, Y)
		return mat.Norm(diff, 2)
	}
	for i := 0; i < int(k.maxIterations); i++ {
		assignCluster(X, centroids, classes, l2_dist)
		updateCentroid(X, centroids, classes)
	}

	return &trainedKmeans{
		centroids: centroids,
	}
}

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

func NewVanilaKmeans(nClusters uint, tolerance float64, maxIterations uint) Kmeans {
	return &vanilaKmeans{
		nClusters:     nClusters,
		tolerance:     tolerance,
		maxIterations: maxIterations,
	}
}

func NewMiniBatchKmeans(nClusters uint, batchSize uint) Kmeans {
	return &miniBatchKmeans{
		nClusters: nClusters,
		batchSize: batchSize,
	}
}

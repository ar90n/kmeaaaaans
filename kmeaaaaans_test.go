package kmeaaaaans

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestClustering(t *testing.T) {
	kmeans := NewVanilaKmeans(2, 1e-4, 10, 1024, KmeansPlusPlus)

	X := mat.NewDense(8, 2, []float64{1, 1, 1, 0, 0, 1, 0, 0, 5, 5, 5, 6, 6, 5, 6, 6})
	trained := kmeans.Fit(X)

	centroids := trained.Centroids()
	if !mat.EqualApprox(centroids, mat.NewDense(2, 2, []float64{0.5, 0.5, 5.5, 5.5}), 1e-10) {
		t.Errorf("trained.Centroids() = %v, want %v", centroids, mat.NewDense(2, 2, []float64{0.5, 0.5, 5.5, 5.5}))
	}

	classes := trained.Predict(X)
	if reflect.DeepEqual(classes, []uint{0, 0, 0, 0, 0, 1, 1, 1}) {
		t.Errorf("trained.Predict(X) = %v, want %v", classes, []uint{0, 0, 0, 0, 0, 1, 1, 1})
	}
}

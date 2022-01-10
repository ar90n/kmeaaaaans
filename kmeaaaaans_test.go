package kmeaaaaans

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestClustering(t *testing.T) {
	for _, kmeans := range []Kmeans{
		NewLloydKmeans(2, 1e-8, 10, 1024, KmeansPlusPlus),
		NewLloydKmeans(2, 1e-8, 10, 2, KmeansPlusPlus),
		NewMiniBatchKmeans(2, 1e-8, 10, 10, 1024, KmeansPlusPlus),
		NewMiniBatchKmeans(2, 1e-8, 10, 10, 4, KmeansPlusPlus),
	} {
		X := mat.NewDense(8, 2, []float64{1, 1, 1, 0, 0, 1, 0, 0, 5, 5, 5, 6, 6, 5, 6, 6})
		trained, _ := kmeans.Fit(X)

		centroids := trained.Centroids()
		expect0 := mat.NewDense(2, 2, []float64{0.5, 0.5, 5.5, 5.5})
		expect1 := mat.NewDense(2, 2, []float64{5.5, 5.5, 0.5, 0.5})
		if !mat.EqualApprox(centroids, expect0, 1e-4) && !mat.EqualApprox(centroids, expect1, 1e-1) {
			t.Errorf("trained.Centroids() = %v, want %v", centroids, mat.NewDense(2, 2, []float64{0.5, 0.5, 5.5, 5.5}))
		}

		classes := trained.Predict(X)
		expect2 := []uint{0, 0, 0, 0, 1, 1, 1, 1}
		expect3 := []uint{1, 1, 1, 1, 0, 0, 0, 0}
		if !reflect.DeepEqual(classes, expect2) && !reflect.DeepEqual(classes, expect3) {
			t.Errorf("trained.Predict(X) = %v, want %v", classes, []uint{0, 0, 0, 0, 1, 1, 1, 1})
		}
	}
}

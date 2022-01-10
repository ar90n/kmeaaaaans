package kmeaaaaans

import (
	"math/rand"
	"runtime"

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

func (k *miniBatchKmeans) Fit(X *mat.Dense) TrainedKmeans {
	nSamples, featDim := X.Dims()
	nextCentroids := calcInitialCentroids(X, k.nClusters, k.initAlgorithm)
	centroids := mat.NewDense(int(k.nClusters), featDim, nil)

	classes := make([]uint, X.RawMatrix().Rows)

	type AssignClusterHandlerParams struct {
		X         *mat.Dense
		Centroids *mat.Dense
		Classes   []uint
		Indices   []uint
	}

	defer ants.Release()
	pool, _ := ants.NewPool(runtime.NumCPU())

	//accSamples := mat.NewDense(int(k.nClusters), featDim, nil)
	accNSamplesInCluster := make([]uint, k.nClusters)
	nSamplesInCluster := make([]uint, k.nClusters)
	indices := makeBatchIndice(uint(X.RawMatrix().Rows), k.batchSize)
	chunks := makeChunks(indices, 256)
	minInertia := float64(1e10)
	minRuns := 0
	//fmt.Println("------------")
	for i := 0; i < int(k.maxIterations) && k.tolerance < calcError(centroids, nextCentroids); i++ {
		for j := 0; j < len(indices); j++ {
			indices[j] = uint(rand.Intn(int(nSamples)))
		}
		for j := 0; j < len(chunks); j++ {
			for k := 0; k < len(chunks[j]); k++ {
				chunks[j][k] = indices[j*len(chunks[j])+k]
			}
		}

		centroids, nextCentroids = nextCentroids, centroids
		nextCentroids.Zero()

		channel := make([]chan float64, 0)
		for _, chunk := range chunks {
			//wg.Add(1)
			chunk := chunk
			channel = append(channel, make(chan float64, 1))
			cur := channel[len(channel)-1]
			//params := AssignClusterHandlerParams{
			//	X:         X,
			//	Centroids: centroids,
			//	Classes:   classes,
			//	Indices:   chunk,
			//}
			//assignClusterHandler.Invoke(params)
			pool.Submit(func() {
				ret := assignCluster(X, centroids, classes, chunk, calcL2Distance)
				//fmt.Println(ret)
				cur <- ret
				close(cur)
			})
		}

		inertia := 0.0
		for _, c := range channel {
			for v := range c {
				inertia += v
				//fmt.Println(inertia)
			}
		}
		if inertia < minInertia {
			minInertia = inertia
			minRuns = 0
		} else {
			minRuns++
		}
		//fmt.Println(i, inertia, minInertia, minRuns)
		if minRuns > 10 {
			break
		}
		//	var wg sync.WaitGroup
		//	assignClusterHandler, _ := ants.NewPoolWithFunc(runtime.NumCPU(), func(i interface{}) {
		//		defer wg.Done()
		//		param := i.(AssignClusterHandlerParams)
		//		assignCluster(param.X, param.Centroids, param.Classes, param.Indices, calcL2Distance)
		//	})
		//	defer assignClusterHandler.Release()
		//
		//		}
		//		wg.Wait()

		for i := 0; i < len(nSamplesInCluster); i++ {
			nSamplesInCluster[i] = 0
		}

		for _, i := range indices {
			cluster := int(classes[i])
			nSamplesInCluster[cluster]++
			accNSamplesInCluster[cluster]++
			centroidData := nextCentroids.RawRowView(cluster)
			featData := X.RawRowView(int(i))
			for j := 0; j < len(featData); j++ {
				centroidData[j] += featData[j]
			}
		}
		//matPrint(nextCentroids)
		for k := 0; k < len(nSamplesInCluster); k++ {
			if 0 < nSamplesInCluster[k] {
				scale := float64(nSamplesInCluster[k]) / float64(accNSamplesInCluster[k])
				scale2 := 1.0 / float64(accNSamplesInCluster[k])
				//if i == 0 {
				//	scale = 1.0 / float64(accNSamplesInCluster[k])
				//}
				centroidData := nextCentroids.RawRowView(k)
				oldCentroidData := centroids.RawRowView(k)
				//fmt.Println(k, centroidData)
				//fmt.Println(k, nSamplesInCluster[k], scale, scale2)
				for j := 0; j < nextCentroids.RawMatrix().Cols; j++ {
					centroidData[j] = scale2*centroidData[j] + (1-scale)*oldCentroidData[j]
				}
			} else {
				nextCentroids.SetRow(k, centroids.RawRowView(k))
			}
		}
		//fmt.Println(nSamplesInCluster)
		//fmt.Println(accNSamplesInCluster)
		//matPrint(centroids)
		//matPrint(nextCentroids)

		//		for _, ii := range indices {
		//			cluster := int(classes[ii])
		//			//nSamplesInCluster[cluster]++
		//			//accNSamplesInCluster[cluster]++
		//			accNSamplesInCluster[cluster] += uint(1 + i)
		//			//centroidData := nextCentroids.RawRowView(cluster)
		//			centroidData := accSamples.RawRowView(cluster)
		//			featData := X.RawRowView(int(ii))
		//			for j := 0; j < len(featData); j++ {
		//				//centroidData[j] += featData[j]
		//				centroidData[j] += float64(1+i) * featData[j]
		//			}
		//		}
		//		//fmt.Println(accNSamplesInCluster)
		//
		//		//for ii := 0; ii < len(nSamplesInCluster); ii++ {
		//		for ii := 0; ii < len(accNSamplesInCluster); ii++ {
		//			//if 0 < nSamplesInCluster[ii] {
		//			if 0 < accNSamplesInCluster[ii] {
		//				scale := 1.0 / float64(accNSamplesInCluster[ii])
		//				centroidData := nextCentroids.RawRowView(ii)
		//				accData := accSamples.RawRowView(ii)
		//				for j := 0; j < featDim; j++ {
		//					centroidData[j] = scale * accData[j]
		//				}
		//				//for j := 0; j < nextCentroids.RawMatrix().Cols; j++ {
		//				//	nextCentroids.Set(ii, j, nextCentroids.At(ii, j)+centroids.At(ii, j)*float64(accNSamplesInCluster[ii]))
		//				//}
		//				//accNSamplesInCluster[ii] += nSamplesInCluster[ii]
		//				//scale := 1.0 / float64(accNSamplesInCluster[ii])
		//				//for j := 0; j < nextCentroids.RawMatrix().Cols; j++ {
		//				//	nextCentroids.Set(ii, j, nextCentroids.At(ii, j)*scale)
		//				//}
		//			} else {
		//				nextCentroids.SetRow(ii, centroids.RawRowView(ii))
		//			}
		//		}
		//matPrint(nextCentroids)
		//if 2 < i {
		//	break
		//}
		//fmt.Println(i, calcError(centroids, nextCentroids))
	}
	centroids = nextCentroids

	return &trainedKmeans{
		centroids: centroids,
	}

}

# kmeaaaaans
Simple kmeans clustering algorithm implementation with pure Go.

## Installation
```bash
$ go get github.com/ar90n/kmeaaaaans
```

## How to use
```go
package main

import (
    "fmt"

    "github.com/ar90n/kmeaaaaans"
    "gonum.org/v1/gonum/mat"
)

func main() {
    X := mat.NewDense(8, 2, []float64{1, 1, 1, 0, 0, 1, 0, 0, 5, 5, 5, 6, 6, 5, 6, 6})

    kmeans := kmeaaaaans.NewLloydKmeans(2, 1e-8, 10, 1024, kmeaaaaans.KmeansPlusPlus)
    trained, _ := kmeans.Fit(X)

    centroids := trained.Centroids()
    classes := trained.Predict(X)

    fmt.Println(centroids)
    fmt.Println(classes)
}
```
## Benchmark

## License
This software is licensed under the Apache License, Version2.0. See ./LICENSE
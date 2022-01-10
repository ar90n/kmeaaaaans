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
### Environment
* CPU: Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz
* MEMORY: 6.79GB

### Dataset
|dataset| source |
|---|:---:|
|s1|[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)|
|s2|[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)|
|s3|[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)|
|s4|[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)|
|mnist|[Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)|
### Result
#### mnist
|implement|duration_ns|inertia|adjusted_mutual_info|adjusted_rand|allocs|completeness|homogeneity|memory_bytes|v_measure|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|kmeaaaaans_lloyd|2073569471|15786137.808150226|-|-|427|-|-|1077152|-|
|kmeaaaaans_minibatch|430284446|15843683.404225197|-|-|149|-|-|1038733|-|
|sklearn_lloyd|7988622426|15634214.873191446|-|-|-|-|-|-|-|
|sklearn_minibatch|378714770|15859564.27098149|-|-|-|-|-|-|-|


#### s1
|implement|duration_ns|inertia|adjusted_mutual_info|adjusted_rand|allocs|completeness|homogeneity|memory_bytes|v_measure|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|kmeaaaaans_lloyd|36232986|200565710.92146432|0.9384505078218724|0.8543047434326204|183|0.957655223959463|0.9211733052140836|872813|0.9390600730321214|
|kmeaaaaans_minibatch|24801494|200960553.45283315|0.9383178276377598|0.8544959163972522|265|0.9575721981448038|0.9209971402747897|872487|0.9389286177107021|
|sklearn_lloyd|263421922|160829158.47744548|0.9577392344404378|0.9143440693645136|-|0.9948182948165927|0.9240807177726164|-|0.9581456844745858|
|sklearn_minibatch|31958192|161482821.17930794|0.9567810479227407|0.9104269713309593|-|0.9947325274000741|0.9223895364146473|-|0.957196093168326|


#### s2
|implement|duration_ns|inertia|adjusted_mutual_info|adjusted_rand|allocs|completeness|homogeneity|memory_bytes|v_measure|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|kmeaaaaans_lloyd|44147153|196206561.3180315|0.9387069390399666|0.913601155077238|236|0.9709890482428535|0.9096166283110391|876219|0.9393014125194773|
|kmeaaaaans_minibatch|25795179|196877207.1500109|0.9383944778029034|0.9101229343530801|301|0.9714196186021895|0.9086571226809943|873978|0.938990772209102|
|sklearn_lloyd|581901043|195823357.37680343|0.9416273562170373|0.9059487094023926|-|0.9768402597839365|0.9099142582860312|-|0.9421902764059243|
|sklearn_minibatch|32696843|197086010.2104493|0.9445431039527143|0.917355161648106|-|0.9763396457492672|0.9157635118450026|-|0.9450818991092016|


#### s3
|implement|duration_ns|inertia|adjusted_mutual_info|adjusted_rand|allocs|completeness|homogeneity|memory_bytes|v_measure|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|kmeaaaaans_lloyd|51246004|225125231.1671057|0.9100484081460027|0.8607698180004462|280|0.9470132272433684|0.8774640399752139|879804|0.9109130239301323|
|kmeaaaaans_minibatch|26347426|231037121.79743654|0.8944987286749871|0.8449868169269585|314|0.9279372260739442|0.8652866855366189|874642|0.8955175330161905|
|sklearn_lloyd|836674898|223897258.70423615|0.8984366303440723|0.853069328462193|-|0.9362786023439942|0.865336808354059|-|0.8994109760291545|
|sklearn_minibatch|35821437|226567169.5165872|0.8925821528689937|0.8383267208332065|-|0.9309695715584747|0.859136303840799|-|0.893611677039608|


#### s4
|implement|duration_ns|inertia|adjusted_mutual_info|adjusted_rand|allocs|completeness|homogeneity|memory_bytes|v_measure|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|kmeaaaaans_lloyd|48078783|209435418.48114234|0.874201040242448|0.8194968780885233|247|0.9112007439503567|0.8423249097576251|876484|0.8754101575828818|
|kmeaaaaans_minibatch|25455592|214927616.66683713|0.867554419196954|0.8116674227670324|288|0.9013630517354568|0.838578542753332|873437|0.868838030897348|
|sklearn_lloyd|953384131|209038429.89043334|0.8645483805338074|0.8029798461424996|-|0.9032456036933398|0.8314169488187811|-|0.8658441410049948|
|sklearn_minibatch|34238427|217289977.8259685|0.8600024207148045|0.7900461057938077|-|0.8991174535058484|0.8266093824179771|-|0.861340169825695|




## License
This software is licensed under the Apache License, Version2.0. See ./LICENSE
package main

import (
	"ar90n/kmeaaaaans"
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/urfave/cli/v2"
	"gonum.org/v1/gonum/mat"
)

type BenchmarkResult struct {
	DurationNS  uint   `json:"duration_ns"`
	MemoryBytes uint   `json:"memory_bytes"`
	Allocs      uint64 `json:"allocs"`
}

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func parseFromSeparateFloat64(str string, del string) ([]float64, error) {
	var floats []float64
	for _, s := range strings.Split(str, del) {
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		floats = append(floats, f)
	}
	return floats, nil
}

func dumpAsSeparatedFloat64(w io.Writer, X *mat.Dense, delimiter string) error {
	nSamples, featDim := X.Dims()

	for i := 0; i < nSamples; i++ {
		for j := 0; j < featDim; j++ {
			if j != 0 {
				_, err := fmt.Fprint(w, delimiter)
				if err != nil {
					return err
				}
			}
			_, err := fmt.Fprint(w, X.At(i, j))
			if err != nil {
				return err
			}
		}
		_, err := fmt.Fprint(w, "\n")
		if err != nil {
			return err
		}
	}

	return nil
}

func readFeatures(r io.Reader, delimiter string) (*mat.Dense, error) {
	scanner := bufio.NewScanner(r)
	scanner.Scan()
	if err := scanner.Err(); err != nil {
		return nil, err
	}

	featBuffer, err := parseFromSeparateFloat64(scanner.Text(), delimiter)
	if err != nil {
		return nil, err
	}
	featDim := len(featBuffer)

	for scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return nil, err
		}

		buf, err := parseFromSeparateFloat64(scanner.Text(), delimiter)
		if err != nil {
			return nil, err
		}
		if len(buf) != featDim {
			return nil, fmt.Errorf("feature dimension mismatch: %d != %d", len(buf), featDim)
		}
		featBuffer = append(featBuffer, buf...)
	}

	nSamples := len(featBuffer) / featDim
	X := mat.NewDense(nSamples, featDim, featBuffer)
	return X, nil
}

func trainAction(c *cli.Context) error {
	nClusters := c.Uint("clusters")
	tolerance := c.Float64("tolerance")
	maxIter := c.Uint("max-iter")
	batchSize := c.Uint("batch-size")
	delimiter := c.String("delimiter")
	initAlgorithm, err := kmeaaaaans.InitAlgorithmFrom(c.String("init-algorithm"))
	if err != nil {
		return err
	}

	kmeans := kmeaaaaans.NewVanilaKmeans(nClusters, tolerance, maxIter, batchSize, initAlgorithm)

	X, err := readFeatures(os.Stdin, delimiter)
	if err != nil {
		return err
	}

	trained := kmeans.Fit(X)
	centroids := trained.Centroids()
	if err := dumpAsSeparatedFloat64(os.Stdout, centroids, delimiter); err != nil {
		return err
	}

	return nil
}

func predictAction(c *cli.Context) error {
	if c.NArg() != 1 {
		return fmt.Errorf("expected 1 argument, got %d", c.NArg())
	}

	delimiter := c.String("delimiter")
	centroidsFilePath := c.Args().First()

	fp, err := os.Open(centroidsFilePath)
	if err != nil {
		return err
	}
	defer fp.Close()

	centroids, err := readFeatures(fp, delimiter)
	if err != nil {
		return err
	}

	kmeans := kmeaaaaans.NewTrainedKmeans(centroids)
	X, err := readFeatures(os.Stdin, delimiter)
	if err != nil {
		return err
	}
	predicts := kmeans.Predict(X)
	for _, p := range predicts {
		fmt.Println(p)
	}

	return nil
}

func benchmarkAction(c *cli.Context) error {
	nClusters := c.Uint("clusters")
	tolerance := c.Float64("tolerance")
	maxIter := c.Uint("max-iter")
	batchSize := c.Uint("batch-size")
	delimiter := c.String("delimiter")
	useJsonFormat := c.Bool("json")
	initAlgorithm, err := kmeaaaaans.InitAlgorithmFrom(c.String("init-algorithm"))
	if err != nil {
		return err
	}

	kmeans := kmeaaaaans.NewVanilaKmeans(nClusters, tolerance, maxIter, batchSize, initAlgorithm)
	X, err := readFeatures(os.Stdin, delimiter)
	if err != nil {
		return err
	}

	result := testing.Benchmark(func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kmeans.Fit(X)
		}
	})

	if useJsonFormat {
		b := BenchmarkResult{
			DurationNS:  uint(result.NsPerOp()),
			MemoryBytes: uint(result.AllocedBytesPerOp()),
			Allocs:      uint64(result.AllocsPerOp()),
		}
		jsonBytes, err := json.Marshal(b)
		if err != nil {
			return err
		}
		fmt.Println(string(jsonBytes))
	} else {
		fmt.Println(result, result.MemString())
	}
	return nil
}

func main() {
	app := &cli.App{
		Name:     "kmeaaaaans",
		HelpName: "kmeaaaaans",
		Usage:    "example program to test kmeaaaaans",
		Commands: []*cli.Command{
			{
				Name:      "train",
				Usage:     "train kmeans",
				UsageText: "kmeaaaaans train [command options]",
				Action:    trainAction,
				Flags: []cli.Flag{
					&cli.UintFlag{
						Name:        "clusters",
						Usage:       "number of clusters",
						Value:       8,
						DefaultText: "8",
					},
					&cli.UintFlag{
						Name:        "max-iter",
						Usage:       "max number of iterations",
						Value:       300,
						DefaultText: "300",
					},
					&cli.Float64Flag{
						Name:        "tolerance",
						Usage:       "tolerance",
						Value:       1e-4,
						DefaultText: "1e-4",
					},
					&cli.UintFlag{
						Name:        "batch-size",
						Usage:       "batch size",
						Value:       1024,
						DefaultText: "1024",
					},
					&cli.StringFlag{
						Name:        "init-algorithm",
						Usage:       "initialization algorithm",
						Value:       "kmeans++",
						DefaultText: "kmeans++",
					},
					&cli.StringFlag{
						Name:        "delimiter",
						Usage:       "delimiter",
						Value:       ",",
						DefaultText: ",",
					},
				},
			},
			{
				Name:      "predict",
				Usage:     "predict classes",
				Action:    predictAction,
				ArgsUsage: "<path to centroids-file>",
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:        "delimiter",
						Usage:       "delimiter",
						Value:       ",",
						DefaultText: ",",
					},
				},
			},
			{
				Name:      "benchmark",
				Usage:     "benchmark",
				UsageText: "kmeaaaaans benchmark [command options]",
				Action:    benchmarkAction,
				Flags: []cli.Flag{
					&cli.UintFlag{
						Name:        "clusters",
						Usage:       "number of clusters",
						Value:       8,
						DefaultText: "8",
					},
					&cli.UintFlag{
						Name:        "max-iter",
						Usage:       "max number of iterations",
						Value:       300,
						DefaultText: "300",
					},
					&cli.Float64Flag{
						Name:        "tolerance",
						Usage:       "tolerance",
						Value:       1e-4,
						DefaultText: "1e-4",
					},
					&cli.UintFlag{
						Name:        "batch-size",
						Usage:       "batch size",
						Value:       1024,
						DefaultText: "1024",
					},
					&cli.StringFlag{
						Name:        "init-algorithm",
						Usage:       "initialization algorithm",
						Value:       "kmeans++",
						DefaultText: "kmeans++",
					},
					&cli.StringFlag{
						Name:        "delimiter",
						Usage:       "delimiter",
						Value:       ",",
						DefaultText: ",",
					},
					&cli.BoolFlag{
						Name:  "json",
						Usage: "output in json format",
					},
				},
			},
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		log.Fatal(err)
	}
}

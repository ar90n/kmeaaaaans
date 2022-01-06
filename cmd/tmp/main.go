package main

import (
	"ar90n/kmeaaaaans"
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/urfave/cli/v2"
	"gonum.org/v1/gonum/mat"
)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

func parseCommaSeparateFloat64(str string) ([]float64, error) {
	var floats []float64
	for _, s := range strings.Split(str, ",") {
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		floats = append(floats, f)
	}
	return floats, nil
}

func trainAction(c *cli.Context) error {
	nClusters := c.Uint("clusters")
	tolerance := c.Float64("tolerance")
	maxIter := c.Uint("max-iter")
	batchSize := c.Uint("batch-size")
	kmeans := kmeaaaaans.NewVanilaKmeans(nClusters, tolerance, maxIter, batchSize)

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	if err := scanner.Err(); err != nil {
		return err
	}

	featBuffer, err := parseCommaSeparateFloat64(scanner.Text())
	if err != nil {
		return err
	}
	featDim := len(featBuffer)

	for scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return err
		}

		buf, err := parseCommaSeparateFloat64(scanner.Text())
		if err != nil {
			return err
		}
		if len(buf) != featDim {
			return fmt.Errorf("feature dimension mismatch: %d != %d", len(buf), featDim)
		}
		featBuffer = append(featBuffer, buf...)
	}

	nSamples := len(featBuffer) / featDim
	X := mat.NewDense(nSamples, featDim, featBuffer)
	trained := kmeans.Fit(X)
	centroids := trained.Centroids()
	matPrint(centroids)

	return nil
}

func predictAction(c *cli.Context) error {
	fmt.Println("added task: ", c.Args().First())
	return nil
}

func main() {
	app := &cli.App{
		Name:  "kmeaaaaans",
		Usage: "example program to test kmeaaaaans",
		Commands: []*cli.Command{
			{
				Name:  "train",
				Usage: "train kmeans",
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
				},
				Action: trainAction,
			},
			{
				Name:   "predict",
				Usage:  "predict kmeans",
				Action: predictAction,
			},
		},
	}

	err := app.Run(os.Args)
	if err != nil {
		log.Fatal(err)
	}
}

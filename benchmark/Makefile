mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
benchmark_dir := $(dir $(mkfile_path))
project_dir := $(abspath $(benchmark_dir)/..)
cwd := $(shell pwd)

s-originals.zip:
	curl -L -O http://cs.joensuu.fi/sipu/datasets/s-originals.zip

s1-cb.txt: s-originals.zip
	unzip -p $^ $@ > $@

s2-cb.txt: s-originals.zip
	unzip -p $^ $@ > $@

s3-cb.txt: s-originals.zip
	unzip -p $^ $@ > $@

s4-cb.txt: s-originals.zip
	unzip -p $^ $@ > $@

s1.txt:
	curl -L -O http://cs.joensuu.fi/sipu/datasets/s1.txt

s2.txt:
	curl -L -O http://cs.joensuu.fi/sipu/datasets/s2.txt

s3.txt:
	curl -L -O http://cs.joensuu.fi/sipu/datasets/s3.txt

s4.txt:
	curl -L -O http://cs.joensuu.fi/sipu/datasets/s4.txt

s1.csv: s1.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s2.csv: s2.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s3.csv: s3.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s4.csv: s4.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s1-cb.csv: s1-cb.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s2-cb.csv: s2-cb.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s3-cb.csv: s3-cb.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

s4-cb.csv: s4-cb.txt
	cat $^ | awk '{printf("%s,%s\n",$$1,$$2)}' > $@

output_sklearn_vanila_s1.csv performance_sklearn_vanila_s1.json: s1.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm vanila 20 $(cwd)/s1.csv $(cwd)/output_sklearn_vanila_s1.csv > $(cwd)/performance_sklearn_vanila_s1.json

output_sklearn_vanila_s2.csv performance_sklearn_vanila_s2.json: s2.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm vanila 20 $(cwd)/s2.csv $(cwd)/output_sklearn_vanila_s2.csv > $(cwd)/performance_sklearn_vanila_s2.json

output_sklearn_vanila_s3.csv performance_sklearn_vanila_s3.json: s3.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm vanila 20 $(cwd)/s3.csv $(cwd)/output_sklearn_vanila_s3.csv > $(cwd)/performance_sklearn_vanila_s3.json

output_sklearn_vanila_s4.csv performance_sklearn_vanila_s4.json: s4.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm vanila 20 $(cwd)/s4.csv $(cwd)/output_sklearn_vanila_s4.csv > $(cwd)/performance_sklearn_vanila_s4.json

output_sklearn_minibatch_s1.csv performance_sklearn_minibatch_s1.json: s1.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm minibatch 20 $(cwd)/s1.csv $(cwd)/output_sklearn_minibatch_s1.csv > $(cwd)/performance_sklearn_minibatch_s1.json

output_sklearn_minibatch_s2.csv performance_sklearn_minibatch_s2.json: s2.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm minibatch 20 $(cwd)/s2.csv $(cwd)/output_sklearn_minibatch_s2.csv > $(cwd)/performance_sklearn_minibatch_s2.json

output_sklearn_minibatch_s3.csv performance_sklearn_minibatch_s3.json: s3.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm minibatch 20 $(cwd)/s3.csv $(cwd)/output_sklearn_minibatch_s3.csv > $(cwd)/performance_sklearn_minibatch_s3.json

output_sklearn_minibatch_s4.csv performance_sklearn_minibatch_s4.json: s4.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm minibatch 20 $(cwd)/s4.csv $(cwd)/output_sklearn_minibatch_s4.csv > $(cwd)/performance_sklearn_minibatch_s4.json

metrics_sklearn_vanila_s1.json: s1.csv s1-cb.csv output_sklearn_vanila_s1.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s1.csv $(cwd)/output_sklearn_vanila_s1.csv --gt-path $(cwd)/s1-cb.csv > $(cwd)/metrics_sklearn_vanila_s1.json

metrics_sklearn_vanila_s2.json: s2.csv s2-cb.csv output_sklearn_vanila_s2.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s2.csv $(cwd)/output_sklearn_vanila_s2.csv --gt-path $(cwd)/s2-cb.csv > $(cwd)/metrics_sklearn_vanila_s2.json

metrics_sklearn_vanila_s3.json: s3.csv s3-cb.csv output_sklearn_vanila_s3.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s3.csv $(cwd)/output_sklearn_vanila_s3.csv --gt-path $(cwd)/s3-cb.csv > $(cwd)/metrics_sklearn_vanila_s3.json

metrics_sklearn_vanila_s4.json: s4.csv s4-cb.csv output_sklearn_vanila_s4.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s4.csv $(cwd)/output_sklearn_vanila_s4.csv --gt-path $(cwd)/s4-cb.csv > $(cwd)/metrics_sklearn_vanila_s4.json

metrics_sklearn_minibatch_s1.json: s1.csv s1-cb.csv output_sklearn_minibatch_s1.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s1.csv $(cwd)/output_sklearn_minibatch_s1.csv --gt-path $(cwd)/s1-cb.csv > $(cwd)/metrics_sklearn_minibatch_s1.json

metrics_sklearn_minibatch_s2.json: s2.csv s2-cb.csv output_sklearn_minibatch_s2.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s2.csv $(cwd)/output_sklearn_minibatch_s2.csv --gt-path $(cwd)/s2-cb.csv > $(cwd)/metrics_sklearn_minibatch_s2.json

metrics_sklearn_minibatch_s3.json: s3.csv s3-cb.csv output_sklearn_minibatch_s3.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s3.csv $(cwd)/output_sklearn_minibatch_s3.csv --gt-path $(cwd)/s3-cb.csv > $(cwd)/metrics_sklearn_minibatch_s3.json

metrics_sklearn_minibatch_s4.json: s4.csv s4-cb.csv output_sklearn_minibatch_s4.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s4.csv $(cwd)/output_sklearn_minibatch_s4.csv --gt-path $(cwd)/s4-cb.csv > $(cwd)/metrics_sklearn_minibatch_s4.json

sklearn_vanila_s1.json: metrics_sklearn_vanila_s1.json performance_sklearn_vanila_s1.json
	jq -s add metrics_sklearn_vanila_s1.json performance_sklearn_vanila_s1.json > $@

sklearn_vanila_s2.json: metrics_sklearn_vanila_s2.json performance_sklearn_vanila_s2.json
	jq -s add metrics_sklearn_vanila_s2.json performance_sklearn_vanila_s2.json > $@

sklearn_vanila_s3.json: metrics_sklearn_vanila_s3.json performance_sklearn_vanila_s3.json
	jq -s add metrics_sklearn_vanila_s3.json performance_sklearn_vanila_s3.json > $@

sklearn_vanila_s4.json: metrics_sklearn_vanila_s4.json performance_sklearn_vanila_s4.json
	jq -s add metrics_sklearn_vanila_s4.json performance_sklearn_vanila_s4.json > $@

sklearn_minibatch_s1.json: metrics_sklearn_minibatch_s1.json performance_sklearn_minibatch_s1.json
	jq -s add metrics_sklearn_minibatch_s1.json performance_sklearn_minibatch_s1.json > $@

sklearn_minibatch_s2.json: metrics_sklearn_minibatch_s2.json performance_sklearn_minibatch_s2.json
	jq -s add metrics_sklearn_minibatch_s2.json performance_sklearn_minibatch_s2.json > $@

sklearn_minibatch_s3.json: metrics_sklearn_minibatch_s3.json performance_sklearn_minibatch_s3.json
	jq -s add metrics_sklearn_minibatch_s3.json performance_sklearn_minibatch_s3.json > $@

sklearn_minibatch_s4.json: metrics_sklearn_minibatch_s4.json performance_sklearn_minibatch_s4.json
	jq -s add metrics_sklearn_minibatch_s4.json performance_sklearn_minibatch_s4.json > $@

s1.json: sklearn_vanila_s1.json sklearn_minibatch_s1.json kmeaaaaans_vanila_s1.json
	jq -s '{"kmeaaaaans_vanila": .[0], "sklearn_vanila": .[1], "sklearn_minibatch": .[2]}' kmeaaaaans_vanila_s1.json sklearn_vanila_s1.json sklearn_minibatch_s1.json > s1.json

s2.json: sklearn_vanila_s2.json sklearn_minibatch_s2.json kmeaaaaans_vanila_s2.json
	jq -s '{"kmeaaaaans_vanila": .[0], "sklearn_vanila": .[1], "sklearn_minibatch": .[2]}' kmeaaaaans_vanila_s2.json sklearn_vanila_s2.json sklearn_minibatch_s2.json > s2.json

s3.json: sklearn_vanila_s3.json sklearn_minibatch_s3.json kmeaaaaans_vanila_s3.json
	jq -s '{"kmeaaaaans_vanila": .[0], "sklearn_vanila": .[1], "sklearn_minibatch": .[2]}' kmeaaaaans_vanila_s3.json sklearn_vanila_s3.json sklearn_minibatch_s3.json > s3.json

s4.json: sklearn_vanila_s4.json sklearn_minibatch_s4.json kmeaaaaans_vanila_s4.json
	jq -s '{"kmeaaaaans_vanila": .[0], "sklearn_vanila": .[1], "sklearn_minibatch": .[2]}' kmeaaaaans_vanila_s4.json sklearn_vanila_s4.json sklearn_minibatch_s4.json > s4.json

kmeaaaaans:
	go build -o kmeaaaaans $(project_dir)/cmd/kmeaaaaans/main.go

performance_kmeaaaaans_vanila_s1.json: s1.csv kmeaaaaans
	cat s1.csv | ./kmeaaaaans benchmark --clusters 20 --json > $@

performance_kmeaaaaans_vanila_s2.json: s2.csv kmeaaaaans
	cat s2.csv | ./kmeaaaaans benchmark --clusters 20 --json > $@

performance_kmeaaaaans_vanila_s3.json: s3.csv kmeaaaaans
	cat s3.csv | ./kmeaaaaans benchmark --clusters 20 --json > $@

performance_kmeaaaaans_vanila_s4.json: s4.csv kmeaaaaans
	cat s4.csv | ./kmeaaaaans benchmark --clusters 20 --json > $@

output_kmeaaaaans_vanila_s1.csv: s1.csv kmeaaaaans
	cat s1.csv | ./kmeaaaaans train --clusters 20 > $@

output_kmeaaaaans_vanila_s2.csv: s2.csv kmeaaaaans
	cat s2.csv | ./kmeaaaaans train --clusters 20 > $@

output_kmeaaaaans_vanila_s3.csv: s3.csv kmeaaaaans
	cat s3.csv | ./kmeaaaaans train --clusters 20 > $@

output_kmeaaaaans_vanila_s4.csv: s4.csv kmeaaaaans
	cat s4.csv | ./kmeaaaaans train --clusters 20 > $@

metrics_kmeaaaaans_vanila_s1.json: s1.csv s1-cb.csv output_kmeaaaaans_vanila_s1.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s1.csv $(cwd)/output_kmeaaaaans_vanila_s1.csv --gt-path $(cwd)/s1-cb.csv > $(cwd)/metrics_kmeaaaaans_vanila_s1.json

metrics_kmeaaaaans_vanila_s2.json: s2.csv s2-cb.csv output_kmeaaaaans_vanila_s2.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s2.csv $(cwd)/output_kmeaaaaans_vanila_s2.csv --gt-path $(cwd)/s2-cb.csv > $(cwd)/metrics_kmeaaaaans_vanila_s2.json

metrics_kmeaaaaans_vanila_s3.json: s3.csv s3-cb.csv output_kmeaaaaans_vanila_s3.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s3.csv $(cwd)/output_kmeaaaaans_vanila_s3.csv --gt-path $(cwd)/s3-cb.csv > $(cwd)/metrics_kmeaaaaans_vanila_s3.json

metrics_kmeaaaaans_vanila_s4.json: s4.csv s4-cb.csv output_kmeaaaaans_vanila_s4.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/s4.csv $(cwd)/output_kmeaaaaans_vanila_s4.csv --gt-path $(cwd)/s4-cb.csv > $(cwd)/metrics_kmeaaaaans_vanila_s4.json

kmeaaaaans_vanila_s1.json: metrics_kmeaaaaans_vanila_s1.json performance_kmeaaaaans_vanila_s1.json
	jq -s add metrics_kmeaaaaans_vanila_s1.json performance_kmeaaaaans_vanila_s1.json > $@

kmeaaaaans_vanila_s2.json: metrics_kmeaaaaans_vanila_s2.json performance_kmeaaaaans_vanila_s2.json
	jq -s add metrics_kmeaaaaans_vanila_s2.json performance_kmeaaaaans_vanila_s2.json > $@

kmeaaaaans_vanila_s3.json: metrics_kmeaaaaans_vanila_s3.json performance_kmeaaaaans_vanila_s3.json
	jq -s add metrics_kmeaaaaans_vanila_s3.json performance_kmeaaaaans_vanila_s3.json > $@

kmeaaaaans_vanila_s4.json: metrics_kmeaaaaans_vanila_s4.json performance_kmeaaaaans_vanila_s4.json
	jq -s add metrics_kmeaaaaans_vanila_s4.json performance_kmeaaaaans_vanila_s4.json > $@

.PHONY: performance_kmeaaaaans_vanila_s_originals
performance_kmeaaaaans_vanila_s_originals: performance_kmeaaaaans_vanila_s1.json performance_kmeaaaaans_vanila_s2.json performance_kmeaaaaans_vanila_s3.json performance_kmeaaaaans_vanila_s4.json

.PHONY: metrics_kmeaaaaans_vanila_s_originals
metrics_kmeaaaaans_vanila_s_originals: metrics_kmeaaaaans_vanila_s1.json metrics_kmeaaaaans_vanila_s2.json metrics_kmeaaaaans_vanila_s3.json metrics_kmeaaaaans_vanila_s4.json

.PHONY: kmeaaaaans_vanila_s_originals
kmeaaaaans_vanila_s_originals: performance_kmeaaaaans_vanila_s_originals metrics_kmeaaaaans_vanila_s_originals

.PHONY: s-originals
s-originals: kmeaaaaans_vanila_s_originals sklearn_vanila_s_originals


MNIST.txt:
	curl -o $@ -L -O http://cs.joensuu.fi/sipu/datasets/MINST.txt

MNIST.csv: MNIST.txt
	cat $^ | tail +2 | tr \  , > $@

output_sklearn_vanila_mnist.csv performance_sklearn_vanila_mnist.json: MNIST.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm vanila 20 $(cwd)/MNIST.csv $(cwd)/output_sklearn_vanila_mnist.csv > $(cwd)/performance_sklearn_vanila_mnist.json

metrics_sklearn_vanila_mnist.json: MNIST.csv output_sklearn_vanila_mnist.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/MNIST.csv $(cwd)/output_sklearn_vanila_mnist.csv > $(cwd)/metrics_sklearn_vanila_mnist.json

sklearn_vanila_mnist.json: performance_sklearn_vanila_mnist.json metrics_sklearn_vanila_mnist.json
	jq -s add performance_sklearn_vanila_mnist.json metrics_sklearn_vanila_mnist.json > $@

output_sklearn_minibatch_mnist.csv performance_sklearn_minibatch_mnist.json: MNIST.csv
	cd $(benchmark_dir) &&\
	poetry run python main.py --algorithm minibatch 20 $(cwd)/MNIST.csv $(cwd)/output_sklearn_minibatch_mnist.csv > $(cwd)/performance_sklearn_minibatch_mnist.json

metrics_sklearn_minibatch_mnist.json: MNIST.csv output_sklearn_minibatch_mnist.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/MNIST.csv $(cwd)/output_sklearn_minibatch_mnist.csv > $(cwd)/metrics_sklearn_minibatch_mnist.json

sklearn_minibatch_mnist.json: performance_sklearn_minibatch_mnist.json metrics_sklearn_minibatch_mnist.json
	jq -s add performance_sklearn_minibatch_mnist.json metrics_sklearn_minibatch_mnist.json > $@

performance_kmeaaaaans_vanila_mnist.json: MNIST.csv kmeaaaaans
	cat MNIST.csv | ./kmeaaaaans benchmark --clusters 10 --json > $@

output_kmeaaaaans_vanila_mnist.csv: MNIST.csv kmeaaaaans
	cat MNIST.csv | ./kmeaaaaans train --clusters 10 > $@

metrics_kmeaaaaans_vanila_mnist.json: MNIST.csv output_kmeaaaaans_vanila_mnist.csv
	cd $(benchmark_dir) &&\
	poetry run python eval.py $(cwd)/MNIST.csv $(cwd)/output_kmeaaaaans_vanila_mnist.csv > $(cwd)/metrics_kmeaaaaans_vanila_mnist.json

kmeaaaaans_vanila_mnist.json: performance_kmeaaaaans_vanila_mnist.json metrics_kmeaaaaans_vanila_mnist.json
	jq -s add performance_kmeaaaaans_vanila_mnist.json metrics_kmeaaaaans_vanila_mnist.json > $@

mnist.json: sklearn_vanila_mnist.json sklearn_minibatch_mnist.json kmeaaaaans_vanila_mnist.json
	jq -s '{"kmeaaaaans_vanila": .[0], "sklearn_vanila": .[1], "sklearn_minibatch": .[2]}' kmeaaaaans_vanila_mnist.json sklearn_vanila_mnist.json sklearn_minibatch_mnist.json > mnist.json

benchmark.json: mnist.json s1.json s2.json s3.json s4.json
	jq -s '{"mnist": .[0], "s1": .[1], "s2": .[2], "s3": .[3], "s4": .[4]}' mnist.json s1.json s2.json s3.json s4.json > $@

report.md: benchmark.json
	cd $(benchmark_dir) &&\
	poetry run python make_report.py $^ > $@
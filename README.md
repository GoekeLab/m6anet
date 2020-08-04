# m6anet

m6anet is a python tool to detect m6a modifications from Nanopore Direct RNA Sequencing data

### Installation

m6anet requires [Python3](https://www.python.org) to run.

To install our xpore package and its dependencies, run

```sh
$ python setup.py install
$ pip install -r requirements.txt 
```

### Detection of differential modification
extract the dataset in this repo.

```sh
$ tar -xvf demo_data.tar.gz .
```

After extraction, you will find 
```
|-- demo_data
    |-- eventalign.txt
    |-- summary.txt
```

First, we need to preprocess the segmented raw signal file in the form of nanopolish eventalign file using 'm6anet-dataprep'.
```sh
$ m6anet-dataprep --eventalign demo_data/eventalign.txt \
--summary demo_data/summary.txt \
--out_dir demo_data --n_processes 4
```
Output files: `eventalign.hdf5`, `eventalign.log`, `inference`, `prepare_for_inference.log`

Now we can run m6anet over our dataestimating differential modification using `m6anet-inference`
```sh
$ m6anet-inference -i dataprep -o . -m m6anet/model/ -n 4
```
Output files: `m6Anet_predictions.csv.gz`
### License
MIT


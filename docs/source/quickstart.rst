.. _quickstart:

Quick Start
==================================
Download and extract the demo dataset from our `S3 bucket <http://s3.ap-southeast-1.amazonaws.com/all-public-data.store.genome.sg/xpore/demo.tar.gz>`_::

    wget http://s3.ap-southeast-1.amazonaws.com/all-public-data.store.genome.sg/xpore/demo.tar.gz
    tar -xvf demo.tar.gz

After extraction, you will find::
    
    demo
    |-- Hek293T_config.yml  # configuration file
    |-- data
        |-- HEK293T-METTL3-KO-rep1  # dataset dir
        |-- HEK293T-WT-rep1 # dataset dir

Each dataset under the ``data`` directory contains the following directories:

* ``fast5`` : Raw signal FAST5 files
* ``fastq`` : Basecalled reads
* ``bamtx`` : Transcriptome-aligned sequence
* ``nanopolish``: Eventalign files obtained from `nanopolish eventalign <https://nanopolish.readthedocs.io/en/latest/quickstart_eventalign.html>`_

Firstly, we need to preprocess the segmented raw signal file in the form of nanopolish eventalign file using 'm6anet-dataprep'::

    m6anet-dataprep --eventalign demo_data/eventalign.txt \
                    --out_dir demo_data --n_processes 4

The output files are stored in ``demo_data``:

* ``data.index``: Indexing of data.json to allow faster access to the file
* ``data.json``: json file containing the features to feed into m6Anet model for prediction
* ``data.log``: Log file containing all the transcripts that have been successfully preprocessed
* ``data.readcount``: File containing the number of reads for each DRACH positions in eventalign.txt
* ``eventalign.index``: Index file created during dataprep to allow faster access of Nanopolish eventalign.txt during dataprep

Now we can run m6anet over our data using m6anet-inference::

    m6anet-run_inference --input_dir demo_data --out_dir demo_data ---n_processes 4

The output files `demo_data/data.result.csv.gz` contains the probability of modification at each individual position for each transcript. The output file will have 4 columns

* ``transcript_id``: The transcript id of the predicted position
* ``transcript_position``: The transcript position of the predicted position
* ``n_reads``: The number of reads for that particular position
* ``probability_modified``: The probability that a given site is modified

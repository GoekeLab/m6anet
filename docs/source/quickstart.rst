.. _quickstart:

Quick Start
==================================
m6Anet requires eventalign.txt from nanopolish::
    nanopolish eventalign --reads reads.fastq --bam reads.sorted.bam --genome transcript.fa --scale-events --signal-index --summary /path/to/summary.txt  --threads 50 > /path/to/eventalign.txt

We have also provided a demo dataset in the repository under /path/to/m6anet/demo/eventalign.txt. 

Firstly, we need to preprocess the segmented raw signal file in the form of nanopolish eventalign file using 'm6anet-dataprep'::

    m6anet-dataprep --eventalign m6anet/demo/eventalign.txt \
                    --out_dir /path/to/output --n_processes 4

The output files are stored in ``/path/to/output``:

* ``data.index``: Indexing of data.json to allow faster access to the file
* ``data.json``: json file containing the features to feed into m6Anet model for prediction
* ``data.log``: Log file containing all the transcripts that have been successfully preprocessed
* ``data.readcount``: File containing the number of reads for each DRACH positions in eventalign.txt
* ``eventalign.index``: Index file created during dataprep to allow faster access of Nanopolish eventalign.txt during dataprep

Now we can run m6anet over our data using m6anet-run_inference::

    m6anet-run_inference --input_dir demo_data --out_dir demo_data --infer_mod-rate --n_processes 4

The output files `demo_data/data.result.csv.gz` contains the probability of modification at each individual position for each transcript. The output file will have 4 columns

* ``transcript_id``: The transcript id of the predicted position
* ``transcript_position``: The transcript position of the predicted position
* ``n_reads``: The number of reads for that particular position
* ``probability_modified``: The probability that a given site is modified
* ``kmer``: The 5-mer motif of a given site
* ``mod_ratio``: The estimated percentage of reads in a given site that is modified

The total run time should not exceed 10 minutes on a normal laptop. We also recommend a threshold of 0.9 for selecting m6A sites
based on the ``probability_modified`` column, which can be relaxed at the expense of having lower model precision.

m6Anet also supports pooling over multiple replicates. To do this, simply input multiple folders containing m6anet-dataprep outputs::
    
        m6anet-run_inference --input_dir demo_data_1 demo_data_2 ... --out_dir demo_data --infer_mod-rate --n_processes 4

.. _quickstart:


**************************
Quick Start
**************************

Dataprep
#######################################
m6Anet dataprep requires eventalign.txt from ``nanopolish eventalign``::

    nanopolish eventalign --reads reads.fastq --bam reads.sorted.bam --genome transcript.fa --scale-events --signal-index --summary /path/to/summary.txt  --threads 50 > /path/to/eventalign.txt

This function segments raw fast5 signals to each position within the transcriptome, allowing m6Anet to predict modification based on the segmented signals. In order to run eventalign, users will need:
* ``reads.fastq``: fastq file generated from basecalling the raw .fast5 files
* ``reads.sorted.bam``: sorted bam file obtained from aligning reads.fastq to the reference transcriptome file
* ``transcript.fa``: reference transcriptome file

We have also provided a demo eventalign.txt dataset in the repository under /path/to/m6anet/m6anet/tests/data/eventalign.txt. Please see `Nanopolish <https://github.com/jts/nanopolish>`_ for more information.

After running nanopolish eventalign, we need to preprocess the segmented raw signal file using 'm6anet dataprep'::

    m6anet dataprep --eventalign /path/to/m6anet/m6anet/tests/data/eventalign.txt \
                    --out_dir /path/to/output --n_processes 4

The output files are stored in ``/path/to/output``:

* ``data.json``: json file containing the features to feed into m6Anet model for prediction
* ``data.log``: Log file containing all the transcripts that have been successfully preprocessed
* ``data.info``: File containing indexing information of data.json for faster file access and the number of reads for each DRACH positions in eventalign.txt
* ``eventalign.index``: Index file created during dataprep to allow faster access of Nanopolish eventalign.txt during dataprep


Inference
#######################################

Once ``m6anet dataprep`` finishes running, we can run ``m6anet inference`` on the dataprep output ::

    m6anet inference --input_dir path/to/output --out_dir path/to/output  --n_processes 4 --num_iterations 1000

m6anet inference will run default human model trained on the HCT116 cell line. In order to run Arabidopsis-based model or the HEK293T-RNA004-based model, please supply the ``--pretrained_model`` argument ::

       ## For the Arabidopsis-based model
       m6anet inference --input_dir path/to/output --out_dir path/to/output  --pretrained_model arabidopsis_RNA002 --n_processes 4 --num_iterations 1000

       ## For the HEK293T-RNA004-based model
       m6anet inference --input_dir path/to/output --out_dir path/to/output  --pretrained_model HEK293T_RNA004 --n_processes 4 --num_iterations 1000

m6Anet will sample 20 reads from each candidate site and average the probability of modification across several round of sampling according to the --num_iterations parameter.
The output file `data.indiv_proba.csv` contains the probability of modification for each read

* ``transcript_id``: The transcript id of the predicted position
* ``transcript_position``: The transcript position of the predicted position
* ``read_index``: The read identifier from nanopolish that corresponds to the actual read_id from nanopolish summary.txt
* ``probability_modified``: The probability that a given read is modified

The output file `data.site_proba.csv` contains the probability of modification at each individual position for each transcript. The output file will have 6 columns

* ``transcript_id``: The transcript id of the predicted position
* ``transcript_position``: The transcript position of the predicted position
* ``n_reads``: The number of reads for that particular position
* ``probability_modified``: The probability that a given site is modified
* ``kmer``: The 5-mer motif of a given site
* ``mod_ratio``: The estimated percentage of reads in a given site that is modified

The mod_ratio column is calculated by thresholding the ``probability_modified`` from `data.indiv_proba.csv` based on the ``--read_proba_threshold`` parameter during ``m6anet inference`` call,
with a default value of 0.033379376 for the default human model HCT116_RNA002 and 0.0032978046219796 for arabidopsis_RNA002 model. We also recommend a threshold of 0.9 to select m6A sites from the ``probability_modified`` column in ``data.site_proba.csv``.
The total run time should not exceed 10 minutes on a normal laptop.


m6Anet also supports pooling over multiple replicates. To do this, simply input multiple folders containing m6anet-dataprep outputs::

        m6anet inference --input_dir data_folder_1 data_folder_2 ... --out_dir output_folder --n_processes 4 --num_iterations 1000

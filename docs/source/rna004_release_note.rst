.. _rna004_release_note:

**************************
Release Note 2.1.0
**************************

``m6anet`` model trained with RNA004 chemistry (development version)
#######################################

The default m6Anet model was trained with the currently available RNA002 direct RNA-Seq kit. Oxford Nanopore is currently providing access to the development version of the next version, RNA004. To make m6A detection possible with RNA004, we now provide an m6Anet model trained on direct RNA
Seq data from the HEK293T cell line using the development version of RNA004. In order to call m6A on data from the RNA004 kit, the following commands can be used:

1) Pre-processing/segmentation/dataprep
Please use f5c with the RNA004 kmer model, as described here:
https://github.com/hasindu2008/f5c/releases/tag/v1.3

The kmer model can be downloaded here:
https://raw.githubusercontent.com/hasindu2008/f5c/v1.3/test/rna004-models/rna004.nucleotide.5mer.model

Then execute eventalign with --kmer-model pointing to the path to the downloaded k-mer model as follows ::

    f5c eventalign --rna -b reads.bam -r reads.fastq -g transciptome.fa -o eventalign.tsv \
    --kmer-model /path/to/rna004.nucleotide.5mer.model --slow5 reads.blow5 --signal-index \
    --scale-events


The output can then be used with m6Anet dataprep (see
https://m6anet.readthedocs.io/en/latest/quickstart.html)

2) Inference
In order to identify m6A from RNA004 data, the RNA004 model has to be specified ::

    m6anet inference --input_dir [INPUT_DIR] --out_dir [OUT_DIR] --pretrained_model HEK293T_RNA004

The RNA004 model is trained on the development version and only underwent limited evaluation on site-level prediction compared to the RNA002 model. The individual read probability accuracy for RNA004 has not been tested. Please report any feedback to us (https://github.com/GoekeLab/m6anet/discussions)

Training and evaluating the RNA004 m6anet
##########################################

We trained m6anet using an RNA004 direct RNA-Seq run of the HEK293T cell line, with m6A positions defined by m6ACE-Seq. We then evaluated the RNA004-based m6anet performance on RNA004 data from the Hek293T and the Hct116 cell line. Here, we used the intersection of all sites identified both in the RNA002 and the RNA004 data to compare the RN004 model (tested on RNA004 data) and the RNA002 model (tested on RNA002 data), using m6ACE-Seq as ground truth (Figure 1-2). The results suggest a comparable performance between the RNA002-trained and the RNA004-trained m6anet.

Please note that the RNA004 will generate higher read numbers, which leads to a higher number of sites being tested.

+------------------------------------------------+----------------------------------------------+
| HEK293T                                        | HCT116                                       |
+=======================+========================+==============================================+
| .. image:: _images/RNA004_mapq20_HEK293T.jpg   | .. image:: _images/RNA004_mapq20_Hct116.jpg  |
+-----------------------+------------------------+----------------------------------------------+

Figure 1: ROC curve comparing the m6Anet model trained on RNA002 and evaluated on RNA002 data with the model trained on RNA004 and evaluated on RNA004. Only sites that were detected in both data sets are used in this comparison. Here, a MAPQ filter of 20 was applied.

+------------------------------------------------+----------------------------------------------+
| HEK293T                                        | HCT116                                       |
+=======================+========================+==============================================+
| .. image:: _images/RNA004_mapq0_HEK293T.jpg    | .. image:: _images/RNA004_mapq0_Hct116.jpg   |
+-----------------------+------------------------+----------------------------------------------+

Figure 2: ROC curve comparing the m6Anet model trained on RNA002 and evaluated on RNA002 data with the model trained on RNA004 and evaluated on RNA004. Only sites that were detected in both data sets are used in this comparison. Here, a MAPQ filter of 0 was applied to the RNA004 data, leading to a higher number of sites which are detected.

The latest RNA004-trained m6anet model is available on https://github.com/goekeLab/m6anet.

Acknowledgments
###########################

We thank Hasindu Gamaarachchi, Hiruna Samarakoon, James Ferguson, and Ira Deveson from the Garvan Institute of Medical Research in Sydney, Australia for enabling the eventalign of the RNA004 data with f5c. We thank Bing Shao Chia, Wei Leong Chew, Arnaud Perrin, Jay Shin, and Hwee Meng Low from the Genome Institute of Singapore for providing the RNA and generating the direct RNA-Seq data, and we thank Paola Florez De Sessions, Lin Yang, Adrien Leger, Lakmal Jayasinghe, Libby Snell, Etienne Raimondeau, and Oxford Nanopore Technologies for providing early access to RNA004, generating the Hek293T data that was used to train the m6Anet model, and for feedback on the results. The model was trained and implemented by Yuk Kei Wan.

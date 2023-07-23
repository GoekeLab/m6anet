.. _release_notes:


**************************
Release Notes
**************************


Release Note 2.1.0
##################


``m6anet`` model trained with RNA004 chemistry (development version)
####################################################################

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

Release Note 2.0.0
##################

API Changes
#######################################

The m6Anet functions for preprocessing, inference, and training have now been simplified. We now provide a single entry point for all m6anet functionalities through the m6anet module. This means
that all the old functionalities of m6Anet are now available through the m6anet module call,
such as ``m6anet dataprep``, ``m6anet inference`` and ``m6anet train`` functions. The command ``m6anet-dataprep``,
``m6anet-run_inference`` and ``m6anet-train`` are deprecated and will be removed in the next version. Please check our updated :ref:`Quickstart page <quickstart>`
and :ref:`Training page <training>` for more details on running m6Anet.

We have also made some changes to the m6anet dataprep function. Previously m6anet-dataprep produces data.index and data.readcount files to run inference,
and we realized that this can be simplified by combining the two files together. The current m6anet dataprep
(and also the deprecated m6anet-dataprep) now produces a single data.info file that combines the information
from both data.index and data.readcount. Furthermore, m6anet inference (also the deprecated m6anet-run_inference) now requires data.info file to be
present in the input directory. We have also provided a function for users to convert older dataprep output files to the newest format using ::

   m6anet convert --input_dir /path/to/old/dataprep/output --out_dir /path/to/old/dataprep/output

This function will create data.info file by combining the old data.index and data.readcount files. The users still need to make sure that data.info file is located in the same folder as data.json file


Faster and Better Inference Implementation
##########################################


In order to minimize the effect of sequencing depth in m6Anet prediction, a fixed number of reads are sampled from each site during m6Anet training.
This process is repeated during inference where the sampling will be repeated several times for each candidate site to stabilize the modification probability.
The number of sampling rounds is controlled through the option `--num_iterations` and the default was set to 5 in the previous version of m6Anet to minimize running time.

\
A low number of sampling iterations results in unstable probability value for individual sites and while the overall performance of m6Anet on large datasets remains unaffected, users looking to identify
and study modifications on individual sites will benefit from a more consistent modification score. In m6Anet 2.0.0, we have improved the inference process so that it can accommodate a higher
number of sampling iterations while still maintaining a relatively fast inference time. Here we include the comparison between the older m6Anet version against the current release in terms of their peak memory usage and running time
over a different number of sampling rounds on our HEK293T dataset with 95030 sites and 8019824 reads. The calculation is done on AMD EPYC 7R32 with `--num_processes` set to 25.

=================================   =====================  ===================  =====================
Version Number                      Peak Memory Usage(MB)  Running Time(s)      Number of Iterations
=================================   =====================  ===================  =====================
m6Anet v-1.1.1                      480.5                  8876.77              50
m6Anet v-1.1.1                      677.9                  18009.92             100
m6Anet v-2.0.0                      553.7                  392.91               5
m6Anet v-2.0.0                      571.3                  229.92               50
m6Anet v-2.0.0                      576.4                  409.71               100
m6Anet v-2.0.0                      578.5                  408.17               1000
=================================   =====================  ===================  =====================

As we can see, the latest version of m6Anet has relatively constant peak memory usage with minimal difference in running time between 100 and 1000 iteration runs. To achieve this, m6Anet
saves each individual read probability file in `data.indiv_proba.csv` before sampling the required amount of reads for each site in parallel. The site level probability is then
saved in `data.site_proba.csv`.


Rounding of Dataprep Output
###########################

Users can now add ``--compress`` flag to ``m6anet dataprep`` to round the dataprep output features to 3 decimal places. In our experience, this reduces the file size for
data.json significantly without compromising model performance.

Arabidopsis Trained m6Anet
##########################

We have also included m6Anet model trained on the Arabidopsis `VIRc dataset <https://elifesciences.org/articles/78808>`_ from our `paper <https://www.nature.com/articles/s41592-022-01666-1>`_ as an option for users who are looking to study
m6A modification on plant genomes or to aggregate predictions from different m6Anet models on their datasets. Here we present single molecular probability results on synthetic RNA from the `curlcake dataset <https://www.nature.com/articles/s41467-019-11713-9>`_

----

.. figure:: _images/m6anet_virc_roc_pr.png
   :align: center
   :alt: VIRc trained m6Anet single-molecular predictions on curlcake dataset.

----

The single-molecule m6A predictions of the Arabidopsis model seem to be comparable with the human model with ROC AUC of 0.89 and PR AUC of 0.90 on the synthetic. We also validate the ability to predict per-molecule
modifications of the Arabidopsis model on the human HEK293T METTL3-KO and wild-type samples that were mixed to achieve an expected relative m6A stoichiometry of 0%, 25%, 50%, 75%, and 100% from `xPore <https://www.nature.com/articles/s41587-021-00949-w>`_
on the sites predicted to be modified in wild-type samples (probability :math:`\geq 0.7`)
) As we can see, from the 1041 shared sites that we inspect across the HEK293T mixtures, the median prediction of the model follows the expected modification ratio.

----

.. figure:: _images/arabidopsis_hek293t_mixtures.png
   :align: center
   :alt: VIRc trained m6Anet single-molecular predictions on HEK293T mixtures dataset.

----

In order to run the Arabidopsis model, please add the following command when running m6anet inference

* ``--read_proba_threshold : 0.0032978046219796``
* ``--model_state_dict : m6anet/m6anet/model/model_states/arabidopsis_virc.pt``
* ``--norm_path : m6anet/m6anet/model/norm_factors/norm_factors_virc.joblib``

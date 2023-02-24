.. _patch_notes:

**************************
Release Note 2.0.0
**************************

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

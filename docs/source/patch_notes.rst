.. _patch_notes:

**************************
Release Note: m6Anet 2.0.0
**************************


Common Entry Point for m6Anet Functions
#######################################

The m6Anet functions for preprocessing, inference and training have now been simplified. We now provide a single entry point for all m6anet functionalities through the m6anet module. This means
that all the old functionalities of m6Anet is now available through the m6anet module call, such as m6anet dataprep, m6anet inference and m6anet train functions. Please check our updated :ref:`Quickstart page <quickstart>`
and :ref:`Training page <training>` for more details.

Faster and Better Inference Implementation
##########################################



In order to minimize the effect of sequencing depth in m6Anet prediction, a fixed number of reads are sampled from each site during m6Anet training.
This process is repeated during inference where the sampling will be repeated several times for each candidate site in order to stabilize the modification probability.
The number of sampling rounds is controlled through the option `--num_iterations` and the default was set to 5 in the previous version of m6Anet in order to minimize running time.

\
Low number of sampling iterations results in unstable probability value for individual sites and while the overall performance of m6Anet on large datasets remains unaffected, users looking to identify
and study modifications on individual sites will benefit from a more consistent modification score. In m6Anet 2.0.0, we have improved the inference process so that it can accomodate higher
number of sampling iterations while still maintaining relatively fast inference time. Here we include the comparison between the older m6Anet version against the current release in terms of their peak memory usage and running time
over different number of sampling rounds on our HEK293T dataset with 95030 sites and 8019824 reads. The calculation is done on AMD EPYC 7R32 with `--num_processes` set to 25.

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

As we can see, the latest version of m6Anet has relatively constant peak memory usage with minimal difference in running time between 100 and 1000 iteration runs.T his is done by saving each individual
read probability in `data.indiv_proba.csv` file before sampling the required amount of reads for each site in parallel.

Arabidopsis Trained m6Anet
##########################

We have also included m6Anet model trained on the Arabidopsis VIRc dataset from our [paper](https://www.nature.com/articles/s41592-022-01666-1) as an option for users who are looking to study
m6A modification on plant genomes or to aggregate predictions from different m6Anet models on their datasets. Here we present single molecular probability results on synthetic RNA from the [curlcake dataset](https://www.nature.com/articles/s41467-019-11713-9)

![alt text](https://github.com/GoekeLab/m6anet/blob/master/figures/m6anet_virc_roc_pr.png "roc_pr_curve")

The single-molecule m6A predictions of the Arabidopsis model seems to be comparable with the human model with ROC AUC of 0.89 and PR AUC of 0.90 on the synthetic. We also validate the ability to predict per-molecule
modifications of the Arabidopsis model on the human HEK293T METTL3-KO and wild-type samples that were mixed to achieve an expected relative m6A stoichiometry of 0%, 25%, 50%, 75%, and 100% from [xPore](https://www.nature.com/articles/s41587-021-00949-w)
on the sites predicted to be modified in wild-type samples (probability :math:`\geq 0.7`)
) As we can see, from the 1041 shared sites that we inspect across the HEK293T mixtures, the median prediction of the model follows the expected modification ratio.

![alt text](https://github.com/GoekeLab/m6anet/blob/master/figures/arabidopsis_hek293t_mixtures.png "hek293t_mixtures")

Here we recommend setting `--read_proba_threshold` of the inference function to 0.0032978046219796 instead of the default.
The arabidopsis model weight can be found in m6anet/m6anet/model/model_states/arabidopsis_virc.pt while the normalization
factors can be found in m6anet/m6anet/model/norm_factors/norm_factors_virc.joblib.

.. _cmd:

Command line arguments
=======================

We provide 2 main scripts to run m6A prediction as the following.

``m6anet-dataprep``
********************

* Input

Output files from ``nanopolish eventalign``

=================================   ==========  ===================  ============================================================================================================
Argument name                       Required    Default value         Description
=================================   ==========  ===================  ============================================================================================================
--eventalign=FILE                   Yes         NA                    Eventalign filepath, the output from nanopolish.         
--out_dir=DIR                       Yes         NA                    Output directory.
--n_processes=NUM                   No          1                     Number of processes to run.
--chunk_size                        No          1000000               chunksize argument for pandas read csv function on the eventalign input
--readcount_max=NUM                 No          1000                  Maximum read counts per gene.
--readcount_min=NUM                 No          1                     Minimum read counts per gene.
--index                             No          True                  To skip indexing the eventalign nanopolish output, can only be used if the index has been created before
--n_neighbors                       No          1                     The number of flanking positions to process
=================================   ==========  ===================  ============================================================================================================

* Output

======================  ==============  ===============================================================================================================================================================
File name               File type       Description
======================  ==============  ===============================================================================================================================================================
eventalign.index        csv             File index indicating the position in the `eventalign.txt` file (the output of nanopolish eventalign) where the segmentation information of each read index is stored, allowing a random access.
data.json               json            Intensity level mean for each position.
data.index              csv             File index indicating the position in the `data.json` file where the intensity level means across positions of each gene is stored, allowing a random access.
data.readcount          csv             Summary of readcounts per gene.
======================  ==============  ===============================================================================================================================================================

``m6anet-run_inference``
************************

* Input

Output files from ``m6anet-dataprep``.

=======================  ==========  ========================= ==============================================================================
Argument name             Required    Default value             Description
=======================  ==========  ========================= ==============================================================================
--input_dir=DIR           Yes         NA                        Input directory that contains data.json, data.index, and data.readcount from m6anet-dataprep
--out_dir=DIR             Yes         NA                        Output directory for the inference results from m6anet
--model_config=FILE       No          prod_pooling.toml         Model architecture specifications. Please see examples in m6anet/model/configs/model_configs/prod_pooling.toml
--model_state_dict=FILE   No          prod_pooling_pr_auc.pt    Model weights to be used for inference. Please see examples in m6anet/model/model_states/
--batch_size              No          64                        Number of sites to be loaded each time for inference
--n_processes=NUM         No          1                         Number of processes to run.
--num_iterations=NUM      No          5                         Number of times m6anet iterates through each potential m6a sites.
=======================  ==========  ========================= ==============================================================================

* Output

======================  ===============     =================================================================================================================================================
File name                File type           Description
======================  ===============     =================================================================================================================================================
data.result.csv.gz      csv.gz              Result table in compressed form  
======================  ===============     =================================================================================================================================================

``m6anet-train``
**************************

====================  ==========  ========================= ==============================================================================
Argument name         Required    Default value             Description
====================  ==========  ========================= ==============================================================================
--model_config=FILE   Yes         NA                        Model architecture specifications. Please see examples in m6anet/model/configs/model_configs/prod_pooling.toml
--train_config=FILE   Yes         NA                        Config file for training the model. Please see examples in m6anet/model/configs/training_configs/oversampled.toml
--save_dir=DIR        Yes         NA                        Save directory to save the training results
--device=STR          No          cpu                       Device to use for training the model. Set to cuda:cuda_id if using GPU
--lr=NUM              No          4e-4                      Learning rate for the ADAM optimizer
--seed=NUM            No          25                        Random seed for model training
--epochs=NUM          No          50                        Number of epochs to train the model.
--num_workers=NUM     No          1                         Number of processes to run.
--save_per_epoch=NUM  No          10                        Number of recurring epoch to save the model
--weight_decay=NUM    No          0                         Weight decay parameteter for the ADAM optimizer
--num_iterations=NUM  No          5                         Number of times m6anet iterates through each potential m6a sites.
====================  ==========  ========================= ==============================================================================

.. _training:

Training m6Anet
=======================
m6Anet expects a training config file and a model config file, both on TOML format. We have provided examples of the model config file and the training config file in:

* m6anet/m6anet/model/configs/model_configs/m6anet.toml
* m6anet/m6anet/model/configs/training_configs/m6anet_train_config.toml

Below is the content of m6anet_train_config.toml

::

    [loss_function]
    loss_function_type = "binary_cross_entropy_loss"

    [dataset]
    root_dir = "/path/to/m6anet-dataprep/output"
    min_reads = 20
    norm_path = "/path/to/m6anet/m6anet/model/norm_factors/norm_dict.joblib"
    num_neighboring_features = 1

    [dataloader]
        [dataloader.train]
        batch_size = 256
        sampler = "ImbalanceOverSampler"

        [dataloader.val]
        batch_size = 256
        shuffle = false

        [dataloader.test]
        batch_size = 256
        shuffle = false

User can modify some basic training information such as the batch_size, the number of neighboring features, as well as the minimum number of reads per site to train m6Anet. We have also calculated the normalization factors required under norm_path variable. In principle, one can even change the loss_function_type by choosing one from m6anet/m6anet/utils/loss_functions.py or defining a new one. Sampler can be set to ImbalanceOverSampler (in which the model will perform oversampling to tackle the data imbalance with m6Anet modification) or any other sampler from m6anet/m6anet/utils/data_utils.py


The training script will look for data.info.labelled file and data.json file under the root_dir directory. While data.info can be obtained by running m6anet dataprep on nanopolish eventalign.txt file, data.info.labelled must be supplied by the user by adding extra columns to the data.info file produced by m6anet dataprep. Additionally, data.info.labelled must be of the following format::

 transcript_id   transcript_position n_reads start end  modification_status set_type
 ENST00000361055 549                 11      0     940  0                   Train
 ENST00000361055 554                 12      940   1969 0                   Train
 ENST00000475035 133                  3      1969  2294 0                   Train
 ENST00000222329 309                 11      2299  3284 0                   Val
 ENST00000222329 2496                15      3284  4593 0                   Val
 ENST00000222329 2631                23      4593  6548 0                   Val
 ENST00000523944 72                   1      6548  6665 0                   Test
 ENST00000523944 2196                14      6665  7853 0                   Test

Here modification status tells the model which positions are modified and which positions are not modified. The column set_type informs the training script which part of the data we should train on and which part of the data should be used for validation and testing purpose. Lastly, n_reads corresponds to the number of reads that comes from the corresponding transcript positions and any sites with n_reads less than the min_reads specified in he training config file will not be used for training validation, or testing. We have also provided an example of data.readcount.labelled in m6anet/demo/ folder.

Below is the content of m6anet.toml::

 model = "prod_sigmoid_pooling"

 [[block]]
 block_type = "DeaggregateNanopolish"
 num_neighboring_features = 1

 [[block]]
 block_type = "KmerMultipleEmbedding"
 input_channel = 66
 output_channel = 2
 num_neighboring_features = 1

 [[block]]
 block_type = "ConcatenateFeatures"

 [[block]]
 block_type = "Linear"
 input_channel = 15
 output_channel = 150
 activation = "relu"
 batch_norm = true

 [[block]]
 block_type = "Linear"
 input_channel = 150
 output_channel = 32
 activation = "relu"
 batch_norm = false

 [[block]]
 block_type = "SigmoidProdPooling"
 input_channel = 32
 n_reads_per_site = 20

The training script will build the model block by block. For additional information on the block type, please check the source code under m6anet/m6anet/model/model_blocks

In order to train m6Anet, please change the root_dir variable inside prod_pooling.toml to m6anet/demo/. Afterwards, run m6anet-dataprep::

     m6anet dataprep --eventalign m6anet/demo/eventalign.txt \
                    --out_dir m6anet/demo/ --n_processes 4

This will produce data.index file and data.json file that will be used for the script to access the preprocessed data Next, to train m6Anet using the demo data, run::

   m6anet train --model_config m6anet/model/configs/model_configs/m6anet.toml --train_config ../m6anet/model/configs/training_configs/m6anet_train_config.toml --save_dir /path/to/save_dir --device cpu --lr 0.0001 --seed 25 --epochs 30 --num_workers 4 --save_per_epoch 1 --num_iterations 5

The model will be trained on cpu for 30 epochs and we will save the model states every 1 epoch. One can replace the device argument with cuda to train with GPU. For complete description of the command line arguments, please see :ref:`Command line arguments page <cmd>`

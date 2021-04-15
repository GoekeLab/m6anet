MODEL_CONFIG_1=/home/christopher/m6anet/m6anet/model/configs/model_configs/1_neighbor/prod_pooling_attention.toml
MODEL_CONFIG_2=/home/christopher/m6anet/m6anet/model/configs/model_configs/1_neighbor/prod_pooling_summary_stats.toml
# TRAIN_CONFIG=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/1_neighbor/oversampled_regularized.toml
TRAIN_CONFIG=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/1_neighbor/normal_regularized.toml
SAVE_DIR_1=/data03/christopher/m6anet_new_dataprep_results/prod_pooling_attention_1_neighbor_normal
SAVE_DIR_2=/data03/christopher/m6anet_new_dataprep_results/prod_pooling_summary_stats_1_neighbor_normal
SETTINGS="--device cuda:2 --epochs 100 --num_workers 10 --save_per_epoch 1 --num_iterations 5"

m6anet-train --model_config $MODEL_CONFIG_1 --train_config $TRAIN_CONFIG --save_dir $SAVE_DIR_1 $SETTINGS
m6anet-train --model_config $MODEL_CONFIG_2 --train_config $TRAIN_CONFIG --save_dir $SAVE_DIR_2 $SETTINGS

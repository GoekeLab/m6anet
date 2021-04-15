MODEL_CONFIG_0=/home/christopher/m6anet/m6anet/model/configs/model_configs/0_neighbor/summary_stats_nn.toml
MODEL_CONFIG_1=/home/christopher/m6anet/m6anet/model/configs/model_configs/1_neighbor/summary_stats_nn.toml
MODEL_CONFIG_2=/home/christopher/m6anet/m6anet/model/configs/model_configs/2_neighbor/summary_stats_nn.toml
MODEL_CONFIG_3=/home/christopher/m6anet/m6anet/model/configs/model_configs/3_neighbor/summary_stats_nn.toml
MODEL_CONFIG_4=/home/christopher/m6anet/m6anet/model/configs/model_configs/4_neighbor/summary_stats_nn.toml
MODEL_CONFIG_5=/home/christopher/m6anet/m6anet/model/configs/model_configs/5_neighbor/summary_stats_nn.toml

TRAIN_CONFIG_0=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/0_neighbor/oversampled_summary_stats.toml
TRAIN_CONFIG_1=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/1_neighbor/oversampled_summary_stats.toml
TRAIN_CONFIG_2=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/2_neighbor/oversampled_summary_stats.toml
TRAIN_CONFIG_3=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/3_neighbor/oversampled_summary_stats.toml
TRAIN_CONFIG_4=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/4_neighbor/oversampled_summary_stats.toml
TRAIN_CONFIG_5=/home/christopher/m6anet/m6anet/model/configs/training_configs/m6a_classification_nanopolish/5_neighbor/oversampled_summary_stats.toml

SAVE_DIR_0=/data03/christopher/m6anet_new_dataprep_results_cv/summary_stats_nn_0_neighbor
SAVE_DIR_1=/data03/christopher/m6anet_new_dataprep_results_cv/summary_stats_nn_1_neighbor
SAVE_DIR_2=/data03/christopher/m6anet_new_dataprep_results_cv/summary_stats_nn_2_neighbor
SAVE_DIR_3=/data03/christopher/m6anet_new_dataprep_results_cv/summary_stats_nn_3_neighbor
SAVE_DIR_4=/data03/christopher/m6anet_new_dataprep_results_cv/summary_stats_nn_4_neighbor
SAVE_DIR_5=/data03/christopher/m6anet_new_dataprep_results_cv/summary_stats_nn_5_neighbor

SETTINGS="--device cuda:2 --epochs 100 --num_workers 10 --save_per_epoch 1 --num_iterations 5 --cv 5"
CV_DIR=/home/christopher/hct116_cv
m6anet-cross_validate --cv_dir $CV_DIR --model_config $MODEL_CONFIG_0 --train_config $TRAIN_CONFIG_0 --save_dir $SAVE_DIR_0 $SETTINGS
m6anet-cross_validate --cv_dir $CV_DIR --model_config $MODEL_CONFIG_1 --train_config $TRAIN_CONFIG_1 --save_dir $SAVE_DIR_1 $SETTINGS
m6anet-cross_validate --cv_dir $CV_DIR --model_config $MODEL_CONFIG_2 --train_config $TRAIN_CONFIG_2 --save_dir $SAVE_DIR_2 $SETTINGS
m6anet-cross_validate --cv_dir $CV_DIR --model_config $MODEL_CONFIG_3 --train_config $TRAIN_CONFIG_3 --save_dir $SAVE_DIR_3 $SETTINGS
m6anet-cross_validate --cv_dir $CV_DIR --model_config $MODEL_CONFIG_4 --train_config $TRAIN_CONFIG_4 --save_dir $SAVE_DIR_4 $SETTINGS
m6anet-cross_validate --cv_dir $CV_DIR --model_config $MODEL_CONFIG_5 --train_config $TRAIN_CONFIG_5 --save_dir $SAVE_DIR_5 $SETTINGS

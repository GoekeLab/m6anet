INPUT_DIR_1=/data03/christopher/m6anet_new_dataprep_hek293/GohGIS_Hek293T_directRNA_Rep2
INPUT_DIR_2=/data03/christopher/m6anet_new_dataprep_hek293/GohGIS_Hek293T-METTLE3-KO-25_directRNA_Rep1-Run1
INPUT_DIR_3=/data03/christopher/m6anet_new_dataprep_hek293/GohGIS_Hek293T-METTLE3-KO-50_directRNA_Rep1-Run1
INPUT_DIR_4=/data03/christopher/m6anet_new_dataprep_hek293/GohGIS_Hek293T-METTLE3-KO-75_directRNA_Rep1-Run1
INPUT_DIR_5=/data03/christopher/m6anet_new_dataprep_hek293/GohGIS_Hek293T-METTL3-KO_directRNA_Rep2_Run1

OUTPUT_DIR_1=$INPUT_DIR_1/prod_pooling_attention_pr_auc
OUTPUT_DIR_2=$INPUT_DIR_2/prod_pooling_attention_pr_auc
OUTPUT_DIR_3=$INPUT_DIR_3/prod_pooling_attention_pr_auc
OUTPUT_DIR_4=$INPUT_DIR_4/prod_pooling_attention_pr_auc
OUTPUT_DIR_5=$INPUT_DIR_5/prod_pooling_attention_pr_auc

MODEL_CONFIG=/home/christopher/m6anet/m6anet/model/configs/model_configs/1_neighbor/prod_pooling_attention.toml
MODEL_WEIGHT=/home/christopher/m6anet/m6anet/model/model_states/attention_pooling_pr_auc.pt
SETTINGS="--model_config $MODEL_CONFIG --model_state_dict $MODEL_WEIGHT --batch_size 512 --num_workers 10 --num_iterations 5 --device cuda:2"

m6anet-run_inference --input_dir $INPUT_DIR_1 --out_dir $OUTPUT_DIR_1 $SETTINGS
m6anet-run_inference --input_dir $INPUT_DIR_2 --out_dir $OUTPUT_DIR_2 $SETTINGS
m6anet-run_inference --input_dir $INPUT_DIR_3 --out_dir $OUTPUT_DIR_3 $SETTINGS
m6anet-run_inference --input_dir $INPUT_DIR_4 --out_dir $OUTPUT_DIR_4 $SETTINGS
m6anet-run_inference --input_dir $INPUT_DIR_5 --out_dir $OUTPUT_DIR_5 $SETTINGS

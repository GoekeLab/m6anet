import pytest
import os
import shutil
import torch
import pandas as pd
import numpy as np
from m6anet.scripts import inference


def test_inference(inference_args, indiv_proba, site_proba):
    inference.main(inference_args)

    test_indiv_proba = os.path.join(inference_args.out_dir, "data.indiv_proba.csv")
    test_site_proba = os.path.join(inference_args.out_dir, "data.site_proba.csv")

    assert(os.path.exists(test_indiv_proba))
    assert(os.path.exists(test_site_proba))

    test_indiv_proba_df = pd.read_csv(test_indiv_proba).sort_values(["transcript_id", "transcript_position", "read_index"])\
        .reset_index(drop=True)
    test_site_proba_df = pd.read_csv(test_site_proba).sort_values(["transcript_id", "transcript_position"])\
        .reset_index(drop=True)

    indiv_proba_df = pd.read_csv(indiv_proba).sort_values(["transcript_id", "transcript_position", "read_index"])\
        .reset_index(drop=True)
    site_proba_df = pd.read_csv(site_proba).sort_values(["transcript_id", "transcript_position"])\
        .reset_index(drop=True)

    assert(np.all(indiv_proba_df["transcript_id"] == test_indiv_proba_df["transcript_id"]))
    assert(np.all(indiv_proba_df["transcript_position"] == test_indiv_proba_df["transcript_position"]))
    assert(np.all(indiv_proba_df["read_index"] == test_indiv_proba_df["read_index"]))
    assert(np.allclose(indiv_proba_df["probability_modified"], test_indiv_proba_df["probability_modified"]))

    assert(np.all(site_proba_df["transcript_id"] == test_site_proba_df["transcript_id"]))
    assert(np.all(site_proba_df["transcript_position"] == test_site_proba_df["transcript_position"]))
    assert(np.allclose(site_proba_df["mod_ratio"], test_site_proba_df["mod_ratio"]))
    assert(np.allclose(site_proba_df["probability_modified"], test_site_proba_df["probability_modified"], atol=1e-2))


def test_inference_replicates(inference_args, indiv_proba, site_proba, data_info, data_json, data_replicate):

    # Create artificial replicates for inference
    shutil.copyfile(data_info, os.path.join(data_replicate, "data.info"))
    shutil.copyfile(data_json, os.path.join(data_replicate, "data.json"))
    inference_args.input_dir = [inference_args.input_dir[0], data_replicate]
    inference.main(inference_args)

    test_indiv_proba = os.path.join(inference_args.out_dir, "data.indiv_proba.csv")
    test_site_proba = os.path.join(inference_args.out_dir, "data.site_proba.csv")

    assert(os.path.exists(test_indiv_proba))
    assert(os.path.exists(test_site_proba))

    # Split by each replicate
    test_indiv_proba_df = pd.read_csv(test_indiv_proba)
    test_site_proba_df = pd.read_csv(test_site_proba)

    test_indiv_proba_df["rep_num"] = test_indiv_proba_df["read_index"].apply(lambda x: x.split("_")[1]).astype(int)
    test_indiv_proba_df["read_index"] = test_indiv_proba_df["read_index"].apply(lambda x: x.split("_")[0]).astype(int)

    test_indiv_proba_0 = test_indiv_proba_df[test_indiv_proba_df["rep_num"] == 0]\
            .sort_values(["transcript_id", "transcript_position", "read_index"])\
            .reset_index(drop=True)
    test_indiv_proba_1 = test_indiv_proba_df[test_indiv_proba_df["rep_num"] == 1]\
        .sort_values(["transcript_id", "transcript_position", "read_index"])\
        .reset_index(drop=True)

    assert(np.all(test_indiv_proba_0["transcript_id"] == test_indiv_proba_1["transcript_id"]))
    assert(np.all(test_indiv_proba_0["transcript_position"] == test_indiv_proba_1["transcript_position"]))
    assert(np.all(test_indiv_proba_0["read_index"] == test_indiv_proba_1["read_index"]))
    assert(np.allclose(test_indiv_proba_0["probability_modified"], test_indiv_proba_1["probability_modified"]))

    indiv_proba_df = pd.read_csv(indiv_proba).rename({'probability_modified': 'gt_probability'}, axis=1)
    site_proba_df = pd.read_csv(site_proba).rename({'probability_modified': 'gt_probability', 'mod_ratio': 'gt_mod_ratio'}, axis=1)

    for test_indiv_proba_df_rep in [test_indiv_proba_0, test_indiv_proba_1]:
        test_df = test_indiv_proba_df_rep.merge(indiv_proba_df, on=["transcript_id", "transcript_position", "read_index"])
        assert(np.allclose(test_df["probability_modified"], test_df["gt_probability"]))

    test_site_proba_df = test_site_proba_df.merge(site_proba_df, on=["transcript_id", "transcript_position"])
    assert(np.allclose(test_site_proba_df["mod_ratio"], test_site_proba_df["gt_mod_ratio"]))
    assert(np.allclose(test_site_proba_df["probability_modified"], test_site_proba_df["gt_probability"], atol=1e-2))

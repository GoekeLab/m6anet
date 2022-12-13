import pandas as pd
import numpy as np
import pytest
import os
import shutil
from m6anet.scripts.dataprep import parallel_index, parallel_preprocess_tx


def test_parallel_index(dataprep_args, tmp_path, eventalign_index):
    eventalign_filepath = dataprep_args['eventalign']
    chunk_size = dataprep_args['chunk_size']
    n_processes = dataprep_args['n_processes']

    parallel_index(eventalign_filepath, chunk_size, tmp_path, n_processes)

    test_fpath = os.path.join(tmp_path, "eventalign.index")

    assert(os.path.exists(test_fpath))

    index_df = pd.read_csv(eventalign_index).sort_values(["transcript_id", "read_index"]).reset_index(drop=True)
    index_test = pd.read_csv(test_fpath).sort_values(["transcript_id", "read_index"]).reset_index(drop=True)

    assert((index_df == index_test).all().all())

@pytest.mark.depends(on=['test_parallel_index'])
def test_parallel_parallel_preprocess_tx(dataprep_args, eventalign_index, data_info, data_json, dataprep_helpers):
    eventalign_filepath = dataprep_args['eventalign']
    out_dir = dataprep_args['out_dir']
    n_processes = dataprep_args['n_processes']
    n_neighbors = dataprep_args['n_neighbors']
    readcount_min = dataprep_args['readcount_min']
    readcount_max = dataprep_args['readcount_max']
    min_segment_count = dataprep_args['min_segment_count']
    n_processes = dataprep_args['n_processes']

    shutil.copy(eventalign_index, out_dir)

    assert(os.path.exists(eventalign_index))

    parallel_preprocess_tx(eventalign_filepath, out_dir, n_processes,
                           readcount_min, readcount_max, n_neighbors,
                           min_segment_count)

    test_data_info = os.path.join(out_dir, "data.info")
    test_data_json = os.path.join(out_dir, "data.json")

    assert(os.path.exists(test_data_info))
    assert(os.path.exists(test_data_json))

    test_data_info_df = pd.read_csv(test_data_info).sort_values(["transcript_id", "transcript_position"])\
        .reset_index(drop=True)
    data_info_df = pd.read_csv(data_info).sort_values(["transcript_id", "transcript_position"])\
        .reset_index(drop=True)

    assert((test_data_info_df["n_reads"] == data_info_df["n_reads"]).all())

    assert(dataprep_helpers.is_equal_data(test_data_info_df, data_info_df, test_data_json, data_json))

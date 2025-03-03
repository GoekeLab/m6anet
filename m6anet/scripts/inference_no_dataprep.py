import os
import torch
import toml
import pkg_resources
import pathlib
import numpy as np
import pandas as pd
import warnings
import subprocess
from itertools import product
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from ..model.model import MILModel
from ..utils.constants import DEFAULT_MODEL_CONFIG,\
    DEFAULT_MIN_READS, DEFAULT_READ_THRESHOLD,\
    DEFAULT_NORM_PATH, PRETRAINED_CONFIGS,\
    DEFAULT_PRETRAINED_MODEL, DEFAULT_PRETRAINED_MODELS
from ..utils.data_nodataprep_utils import NanopolishDS, NanopolishReplicateDS, inference_collate
from ..utils.inference_utils import run_inference
from torch.utils.data import DataLoader


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    # Required arguments
    parser.add_argument("--blow5", nargs="*",
                        help='path to the blow5 file.',
                        required=True)
    parser.add_argument("--fastq", nargs="*",
                        help='path to the fastq file.',
                        required=True)
    parser.add_argument("--bam", nargs="*",
                        help='path to the bam file.',
                        required=True)
    parser.add_argument("--transcript_fasta",
                        help='path to the transcriptome fasta file.',
                        required=True)
    parser.add_argument("--out_dir",
                        help='directory to output inference results.',
                        required=True)

    # Optional arguments
    parser.add_argument("--pretrained_model",
                        help="pre-trained model available at m6anet. Options include {}.".format(DEFAULT_PRETRAINED_MODELS),
                        default=DEFAULT_PRETRAINED_MODEL, type=str)
    parser.add_argument("--model_config",
                        help='path to model config file.',
                        default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model_state_dict",
                        help='path to model weights.',
                        default=None)
    parser.add_argument("--norm_path",
                        help='path to normalization factors file',
                        default=DEFAULT_NORM_PATH)
    parser.add_argument("--batch_size",
                        help='batch size for inference.',
                        default=16, type=int)
    parser.add_argument("--save_per_batch",
                        help='saving inference results every save_per_batch multiples.',
                        default=2, type=int)
    parser.add_argument("--n_processes",
                        help='number of processes to run.',
                        default=25, type=int)
    parser.add_argument("--num_iterations",
                        help='number of sampling run.',
                        default=1000, type=int)
    parser.add_argument("--device",
                        help='device to perform inference with.',
                        default='cpu', type=str)
    parser.add_argument("--seed",
                        help='random seed for sampling.',
                        default=0, type=int)
    parser.add_argument("--read_proba_threshold",
                        help='default probability threshold for a read to be considered modified.',
                        default=DEFAULT_READ_THRESHOLD, type=float)
    return parser

def readFasta(transcript_fasta, is_gff=0):
    fasta=open(transcript_fasta,"r")
    entries,separate_by_pipe="",False
    for ln in fasta:
        entries+=ln
    entries=entries.split(">")
    if len(entries[1].split("|"))>1:
        separate_by_pipe=True
    dict={}
    for entry in entries:
        entry=entry.split("\n")
        if len(entry[0].split()) > 0:
            id=entry[0].split(' ')[0]
            #seq="".join(entry[1:])
            dict[id.split('.')[0]]=id #dict[id]=[seq]
            if is_gff > 0:
                if separate_by_pipe == True:
                    g_id=info[1] #.split(".")[0]
                else:
                    g_id=entry[0].split("gene:")[1] #.split(".")[0]
                dict[id].append(g_id)
    return dict

def main(args):

    blow5 = args.blow5[0]
    fastq = args.fastq[0]
    bam   = args.bam[0]
    fasta = args.transcript_fasta

    if args.model_state_dict is not None:
        warnings.warn("--model_state_dict is specified, overwriting default model weights")
    else:
        if args.pretrained_model not in DEFAULT_PRETRAINED_MODELS:
            raise ValueError("Invalid pretrained model {}, must be one of {}".format(args.pretrained_model, DEFAULT_PRETRAINED_MODELS))

        args.model_state_dict = PRETRAINED_CONFIGS[args.pretrained_model][0]
        args.read_proba_threshold = PRETRAINED_CONFIGS[args.pretrained_model][1]
        args.norm_path = PRETRAINED_CONFIGS[args.pretrained_model][2]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = MILModel(toml.load(args.model_config)).to(args.device)
    model.load_state_dict(torch.load(args.model_state_dict,
                                     map_location=torch.device(args.device)))

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(args.out_dir, "data.site_proba.csv"),'w', encoding='utf-8') as f:
        f.write('transcript_id,transcript_position,n_reads,probability_modified,kmer,mod_ratio\n')
    with open(os.path.join(args.out_dir, "data.indiv_proba.csv"), 'w', encoding='utf-8') as g:
        g.write('transcript_id,transcript_position,read_index,probability_modified\n')

    NUM_NEIGHBORING_FEATURES=1
    CENTER_MOTIFS = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
    FLANKING_MOTIFS = [['G', 'A', 'C', 'T'] for i in range(NUM_NEIGHBORING_FEATURES)]
    ALL_KMERS = list(["".join(x) for x in product(*(FLANKING_MOTIFS + CENTER_MOTIFS + FLANKING_MOTIFS))])
    ALL_KMERS = np.unique(np.array(list(map(lambda x: [x[i:i+5] for i in range(len(x) -4)],
                                        ALL_KMERS))).flatten())
    grep_pattern='"'+'\|'.join(list(ALL_KMERS))+'"'

    cmd='samtools idxstats '+bam+' > stats.txt'
    subprocess.run(cmd, shell=True, check=True)
    fasta_dict=readFasta(fasta)
    for ln in open('stats.txt','r'):
        ln=ln.strip().split('\t')
        if int(ln[2]) > 0 and ln[0].split('.')[0] in fasta_dict:
            tx_id=ln[0]
            cmd='samtools view -b '+bam+' '+tx_id+' > '+tx_id+'.bam'
            subprocess.run(cmd, shell=True, check=True)
            cmd='samtools index '+tx_id+'.bam'
            subprocess.run(cmd, shell=True, check=True)
            try:
                cmd='/home/wanyk/workdir/integrate_f5c/f5c/f5c eventalign -r '+fastq+' -b '+tx_id+'.bam -g '+fasta+' --slow5 '+blow5+' -t 1 --signal-index --m6anet --min-mapq 0 | grep '+grep_pattern+' - > '+tx_id+'_eventalign.txt'
                subprocess.run(cmd, shell=True, check=True)
                subprocess.run('rm '+tx_id+'.bam*',shell=True, check=True)
                eventalign_result=pd.read_csv(tx_id+'_eventalign.txt',sep='\t').iloc[:,1:7]
                eventalign_result.columns=['position','reference_kmer','read_index','event_level_mean','event_stdv','event_length']
                subprocess.run('rm '+tx_id+'_eventalign.txt',shell=True, check=True)
                tmpl=[]
                for pos in sorted(list(set(eventalign_result['position']))):
                    d={tx_id:{}}
                    pos_df=eventalign_result[eventalign_result['position'].isin([pos-1,pos,pos+1])]
                    for read_index in set(pos_df['read_index']):
                        tmp_df=pos_df[pos_df['read_index'] == read_index]
                        if len(tmp_df) == 3:
                            sebenmer=tmp_df['reference_kmer'].iloc[0][0]+tmp_df['reference_kmer'].iloc[1]+tmp_df['reference_kmer'].iloc[2][-1]
                            entry=[tmp_df['event_length'].iloc[0],tmp_df['event_stdv'].iloc[0],tmp_df['event_level_mean'].iloc[0],
                                   tmp_df['event_length'].iloc[1],tmp_df['event_stdv'].iloc[1],tmp_df['event_level_mean'].iloc[1],
                                   tmp_df['event_length'].iloc[2],tmp_df['event_stdv'].iloc[2],tmp_df['event_level_mean'].iloc[2],read_index]
                            if pos not in d[tx_id]:
                                d[tx_id][pos]={}
                                d[tx_id][pos][sebenmer]=[entry]
                            else:
                                d[tx_id][pos][sebenmer].append(entry)
                    if len(d[tx_id]) > 0:
                        if len(d[tx_id][pos][sebenmer]) >= DEFAULT_MIN_READS:
                            tmpl.append(d)

                if len(blow5.split(',')) == 1:
                    ds = NanopolishDS(tmpl, DEFAULT_MIN_READS, args.norm_path, mode='Inference')
                # else:
                #     ds = NanopolishReplicateDS(input_dir, DEFAULT_MIN_READS, args.norm_path, mode='Inference')
                dl = DataLoader(ds, num_workers=args.n_processes, collate_fn=inference_collate, batch_size=args.batch_size,
                                shuffle=False)
                run_inference(model, dl, args)

            except subprocess.CalledProcessError:
                subprocess.run('rm '+tx_id+'*',shell=True, check=True)
                pass

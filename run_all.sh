#!/bin/bash

set -e

python get_protein_seq.py subcellular_localization.tsv.txt ensembl_gene_transcript_protein_map.txt
python do_learning.py
python build_roc_curves.py

#!/bin/bash
# module load gcc/9.2.0 cuda/11.7 python/3.9.14 git/2.9.5
conda deactivate
source synthesis_env/bin/activate
jupyter server --no-browser
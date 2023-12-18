# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/2023-fall/course/hpc/assignments/2023-xue/mat_data
RESPATH=/home/2023-fall/course/hpc/assignments/2023-xue/gemm_res

EXECUTABLE=$1
REP=64

srun -n 1 ${EXECUTABLE} ${REP} ${DATAPATH}/utm5940.csr ${RESPATH}/utm5940.csr

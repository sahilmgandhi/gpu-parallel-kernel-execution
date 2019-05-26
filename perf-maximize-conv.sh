#!/bin/bash

# conv1

nx=(32 64 96 128 160 192 224 256)
ny=(256)
nn=(64)
ni=(64)



nty=(1) #might need to bump to 2, but need to change conv.cu
nby=(32)
ntx=(1) #might need to bump to 2, but need to change conv.cu
nbx=(32)
ntz=(32)

MAX_tii_ti_tnn_tn_tx_ty=""
MAX_PERF=0

echo "RUNNING TILING SCRIPT CONVOLUTION ..."
echo

touch perf-conv-results2.csv
echo "" > perf-conv-results2.csv

for x in ${nx[@]}
do

for y in ${ny[@]}
do

for n in ${nn[@]}
do

for b in ${ni[@]}
do

for i in ${nty[@]}
do
    for j in ${nby[@]}
    do
		for k in ${ntx[@]}
		do
			for l in ${nbx[@]}
			do
                for m in ${ntz[@]}
                do
                        rm ./opt-conv1 || true
                        make -f make-conv opt-conv1 Nx=${x} Ny=${y} Nn=${n} Ni=${b} NUM_THREADS_Y=${i} NUM_BLOCKS_Y=${j} NUM_THREADS_X=${k} NUM_BLOCKS_X=${l} NUM_THREADS_Z=${m}  2>&1 >> /dev/null

                        OUT=`./opt-conv1` 

                        PERF=`echo $OUT | grep "GFlops (MAC=2) 1:" | sed -e 's/.*GFlops (MAC=2) 1: \(.*\) G.*/\1/'`
                        EXEC_TIME=`echo $OUT | grep "elapsed (sec):" | sed -e 's/.*elapsed (sec): \(.*\) G.*G.*/\1/'`
                        
                        PERF2=$PERF # floating point form
                        PERF=$((`printf %.0f $PERF`))

                        if [ $PERF -gt 0 ] && [ $PERF -lt 2000 ]
                        then
                            echo "NTY: $i NBY: $j NTX: $k NBX: $l NTZ: $m  => $PERF GFlops ||| Max: $MAX_PERF GFlops w/ $MAX_NTY_NBY_NTX_NBX_NTZ"

                            (( num_b_z = 64 / $m ))

                            # ./profiling2.sh -a

                            # dram_read=`cat read-profiling.txt | grep "dram_read_throughput" | sed -r "s/.*dram_read_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"`

                            # dram_read_trans=`cat read-profiling.txt | grep "dram_read_transactions" | sed -r "s/.*dram_read_transactions[A-Za-z ]*?([0-9]+).*/\1/"`

                            # l2_tex_read=`cat read-profiling.txt | grep "l2_tex_read_transactions" | sed -r "s/.*\(Texture Reads\)[A-Za-z ]*?([0-9]+).*/\1/"`

                            # dram_write=`cat write-profiling.txt | grep "dram_write_throughput" | sed -r "s/.*dram_write_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"`

                            # dram_write_trans=`cat write-profiling.txt | grep "dram_write_transactions" | sed -r "s/.*dram_write_transactions[A-Za-z ]*?([0-9]+).*/\1/"`

                            # sys_write=`cat write-profiling.txt | grep "sysmem_write_throughput" | sed -r "s/.*sysmem_write_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"`

                            # sys_write_trans=`cat write-profiling.txt | grep "sysmem_write_transactions" | sed -r "s/.*sysmem_write_transactions[A-Za-z ]*?([0-9]+).*/\1/"`

                            echo "$EXEC_TIME,$PERF2,$x,$y,$n,$i,$l,$j,$num_b_z,1,$dram_read,$dram_read_trans,$l2_tex_read,$dram_write,$dram_write_trans,$sys_write,$sys_write_trans" >> perf-conv-results2.csv

                        fi
                        if [ $PERF -gt $MAX_PERF ] && [ $PERF -lt 2000 ]
                        then
                            MAX_PERF=$PERF
                            MAX_NTY_NBY_NTX_NBX_NTZ="NTY: $i NBY: $j NTX: $k NBX: $l NTZ: $m"
                        fi

                done
			done
		done
    done
done
done
done
done
done

echo
echo "MAX PERF => $MAX_PERF GFlops"
echo "MAX_NTY_NBY_NTX_NBX_NTZ => $MAX_NTY_NBY_NTX_NBX_NTZ"



# # Example sed/grep commands

# cat write-profiling.txt | grep "dram_write_throughput" | sed -r "s/.*dram_write_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"
# cat write-profiling.txt | grep "dram_write_transactions" | sed -r "s/.*dram_write_transactions[A-Za-z ]*?([0-9]+).*/\1/"
# cat write-profiling.txt | grep "sysmem_write_throughput" | sed -r "s/.*sysmem_write_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"
# cat write-profiling.txt | grep "sysmem_write_transactions" | sed -r "s/.*sysmem_write_transactions[A-Za-z ]*?([0-9]+).*/\1/"

# cat read-profiling.txt | grep "dram_read_throughput" | sed -r "s/.*dram_read_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"
# cat read-profiling.txt | grep "dram_read_transactions" | sed -r "s/.*dram_read_transactions[A-Za-z ]*?([0-9]+).*/\1/"
# cat read-profiling.txt | grep "l2_tex_read_transactions" | sed -r "s/.*\(Texture Reads\)[A-Za-z ]*?([0-9]+).*/\1/"

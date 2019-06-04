#!/bin/bash

# conv1

nx=(256)
# ny=(256)
nn=(64)
# ni=(64)
nb=(1 3 5 7 9 10)

MAX_PARAMS=""
MAX_PERF=0

echo "RUNNING SCRIPT TO GET VARIOUS CONVOLUTION METRICS ..."
echo

touch conc-batch-res.csv
echo "" > conc-batch-res.csv

touch seq-batch-res.csv
echo "" > seq-batch-res.csv


for i in ${nx[@]}
do
    for j in ${nn[@]}
    do
		for k in ${nb[@]}
		do
            rm ./opt-conv1 || true
            make opt-conv1 NX_PARAM=${i} NY_PARAM=${i} NN_PARAM=${j} NI_PARAM=${j} NUM_BATCHES=${k} 2>&1 >> /dev/null

            OUT=`./opt-conv1` 
            PERF=`echo $OUT | grep "GFlops (MAC=2) 1:" | sed -e 's/.*GFlops (MAC=2) 1: \(.*\) c.*/\1/'`
            EXEC_TIME=`echo $OUT | grep "elapsed (sec):" | sed -e 's/.*elapsed (sec): \(.*\) G.*c.*/\1/'`
            
            PERF2=$PERF # floating point form
            PERF=$((`printf %.0f $PERF`))

            if [ $PERF -gt 0 ] && [ $PERF -lt 2000 ]
            then
                echo "NX: $i NY: $i NN: $j NI: $j NB: $k => $PERF GFlops ||| Max: $MAX_PERF GFlops w/ $MAX_PARAMS"
                echo "$EXEC_TIME,$PERF2,$i,$i,$j,$j,$k" >> seq-batch-res.csv
            fi

            rm ./opt-conv1c || true
            make opt-conv1c NX_PARAM=${i} NY_PARAM=${i} NN_PARAM=${j} NI_PARAM=${j} NUM_BATCHES=${k} 2>&1 >> /dev/null

            OUT=`./opt-conv1c` 
            PERF=`echo $OUT | grep "GFlops (MAC=2) 1:" | sed -e 's/.*GFlops (MAC=2) 1: \(.*\) c.*/\1/'`
            EXEC_TIME=`echo $OUT | grep "elapsed (sec):" | sed -e 's/.*elapsed (sec): \(.*\) G.*c.*/\1/'`
            
            PERF2=$PERF # floating point form
            PERF=$((`printf %.0f $PERF`))

            if [ $PERF -gt 0 ] && [ $PERF -lt 2000 ]
            then
                echo "NX: $i NY: $i NN: $j NI: $j NB: $k => $PERF GFlops ||| Max: $MAX_PERF GFlops w/ $MAX_PARAMS"
                echo "$EXEC_TIME,$PERF2,$i,$i,$j,$j,$k" >> conc-batch-res.csv
            fi

            if [ $PERF -gt $MAX_PERF ] && [ $PERF -lt 2000 ]
            then
                MAX_PERF=$PERF
                MAX_PARAMS="NX: $i NY: $i NN: $j NI: $j NB: $k"
            fi
		done
    done
done

echo
echo "MAX PERF => $MAX_PERF GFlops"
echo "MAX_PARAMS => $MAX_PARAMS"



# # Example sed/grep commands

# cat write-profiling.txt | grep "dram_write_throughput" | sed -r "s/.*dram_write_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"
# cat write-profiling.txt | grep "dram_write_transactions" | sed -r "s/.*dram_write_transactions[A-Za-z ]*?([0-9]+).*/\1/"
# cat write-profiling.txt | grep "sysmem_write_throughput" | sed -r "s/.*sysmem_write_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"
# cat write-profiling.txt | grep "sysmem_write_transactions" | sed -r "s/.*sysmem_write_transactions[A-Za-z ]*?([0-9]+).*/\1/"

# cat read-profiling.txt | grep "dram_read_throughput" | sed -r "s/.*dram_read_throughput[A-Za-z ]*?([0-9]+\.[0-9]*)(KB\/s|MB\/s|GB\/s).*/\1\2/"
# cat read-profiling.txt | grep "dram_read_transactions" | sed -r "s/.*dram_read_transactions[A-Za-z ]*?([0-9]+).*/\1/"
# cat read-profiling.txt | grep "l2_tex_read_transactions" | sed -r "s/.*\(Texture Reads\)[A-Za-z ]*?([0-9]+).*/\1/"

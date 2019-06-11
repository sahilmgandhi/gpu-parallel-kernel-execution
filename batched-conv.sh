#!/bin/bash

# conv1

nx=(64 128 256 512)
# ny=(256)
nn=(16 32 64)
# ni=(64)
nb=(1 2 4 6 8 10 12 14 16 18 20)

MAX_PARAMS=""
MAX_PERF=0

echo "RUNNING SCRIPT TO GET VARIOUS CONVOLUTION METRICS ..."
echo

touch batched-conv.csv
echo "" > batched-conv.csv

for i in ${nx[@]}
do
    for j in ${nn[@]}
    do
        echo "IN${i}OUT${j},IN${i}OUT${j},,,,,BATCH" >> batched-conv.csv
		for k in ${nb[@]}
		do
            rm ./opt-conv-batched || true
            make opt-conv-batched NX_PARAM=${i} NY_PARAM=${i} NN_PARAM=${j} NI_PARAM=${j} NUM_BATCHES=${k} 2>&1 >> /dev/null

            OUT=`./opt-conv-batched` 
            PERF=`echo $OUT | grep "GFlops (MAC=2) 1:" | sed -e 's/.*GFlops (MAC=2) 1: \(.*\) c.*/\1/'`
            EXEC_TIME=`echo $OUT | grep "elapsed (sec):" | sed -e 's/.*elapsed (sec): \(.*\) G.*c.*/\1/'`
            
            PERF2=$PERF # floating point form
            PERF=$((`printf %.0f $PERF`))

            if [ $PERF -ge 0 ] && [ $PERF -le 8000 ]
            then
                echo "Sequential: NX: $i NY: $i NN: $j NI: $j NB: $k => $PERF GFlops ||| Max: $MAX_PERF GFlops w/ $MAX_PARAMS"
                echo "$EXEC_TIME,$PERF2,$i,$i,$j,$j,$k" >> batched-conv.csv
            fi

            if [ $PERF -ge $MAX_PERF ] && [ $PERF -le 8000 ]
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
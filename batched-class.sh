#!/bin/bash

ni=(256 512 1024 2048 4096 12544)
nn=(64 128 256 512 1024 2048 4096)
nb=(1 2 4 6 8 10 12 14 16 18 20 30 40 50 60)

MAX_PARAMS=""
MAX_PERF=0

echo "RUNNING SCRIPT TO GET VARIOUS CLASSIFIER METRICS ..."
echo

touch batched-classifier.csv
echo "" > batched-classifier.csv

for i in ${ni[@]}
do
    for j in ${nn[@]}
    do
        echo "IN${i}OUT${j},IN${i}OUT${j},,,BATCH" >> batched-classifier.csv
        for k in ${nb[@]}
        do
            rm ./opt-class-batched || true
            make opt-class-batched NN_PARAM=${j} NI_PARAM=${i} NUM_BATCHES=${k} 2>&1 >> /dev/null

            OUT=`./opt-class-batched` 
            PERF=`echo $OUT | grep "GFlops (MAC=2) 1:" | sed -e 's/.*GFlops (MAC=2) 1: \(.*\) c.*/\1/'`
            EXEC_TIME=`echo $OUT | grep "elapsed (sec):" | sed -e 's/.*elapsed (sec): \(.*\) G.*c.*/\1/'`
            
            PERF2=$PERF # floating point form
            PERF=$((`printf %.0f $PERF`))

            if [ $PERF -ge 0 ] && [ $PERF -le 6000 ]
            then
                echo "Sequential: NI: $i NN: $j NB: $k => $PERF GFlops ||| Max: $MAX_PERF GFlops w/ $MAX_PARAMS"
                echo "$EXEC_TIME,$PERF2,$i,$j,$k" >> batched-classifier.csv
            fi

            if [ $PERF -ge $MAX_PERF ] && [ $PERF -le 6000 ]
            then
                MAX_PERF=$PERF
                MAX_PARAMS="NN: $j NI: $i NB: $k"
            fi
        done
    done
done

echo
echo "MAX PERF => $MAX_PERF GFlops"
echo "MAX_PARAMS => $MAX_PARAMS"
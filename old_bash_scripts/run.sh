#!/bin/bash

# macOS:           Use `sysctl -n hw.*cpu_max`, which returns the values of
#                  interest directly.
#                  CAVEAT: Using the "_max" key suffixes means that the *maximum*
#                          available number of CPUs is reported, whereas the
#                          current power-management mode could make *fewer* CPUs
#                          available; dropping the "_max" suffix would report the
#                          number of *currently* available ones; see [1] below.
#
# Linux:           Parse output from `lscpu -p`, where each output line represents
#                  a distinct (logical) CPU.
#                  Note: Newer versions of `lscpu` support more flexible output
#                        formats, but we stick with the parseable legacy format
#                        generated by `-p` to support older distros, too.
#                        `-p` reports *online* CPUs only - i.e., on hot-pluggable
#                        systems, currently disabled (offline) CPUs are NOT
#                        reported.

# Number of LOGICAL CPUs (includes those reported by hyper-threading cores)
  # Linux: Simply count the number of (non-comment) output lines from `lscpu -p`,
  # which tells us the number of *logical* CPUs.
logicalCpuCount=$([ $(uname) = 'Darwin' ] &&
                       sysctl -n hw.logicalcpu_max ||
                       lscpu -p | egrep -v '^#' | wc -l)

# Number of PHYSICAL CPUs (cores).
  # Linux: The 2nd column contains the core ID, with each core ID having 1 or
  #        - in the case of hyperthreading - more logical CPUs.
  #        Counting the *unique* cores across lines tells us the
  #        number of *physical* CPUs (cores).
physicalCpuCount=$([ $(uname) = 'Darwin' ] &&
                       sysctl -n hw.physicalcpu_max ||
                       lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)

IMAGE_FILES=(`ls ${PWD}/images/*.jpg`)
IMAGE_COUNT=${#IMAGE_FILES[@]}
OUTPUT_FILE="${PWD}/results.csv"
PROCESS_CNT=$(($((logicalCpuCount > physicalCpuCount ? logicalCpuCount : physicalCpuCount))-1))
MODE="w"

ST=`date +%s`
./build_inference.sh

echo ""
echo "Building parser..."
cd parser
rm -rf $OUTPUT_FILE
make
cd ~-

ET=`date +%s`
echo "Done! (Took $((ET-ST)) seconds for building)"
echo ""

echo "Number of Images: ${IMAGE_COUNT}"
echo "Number of Cores: ${PROCESS_CNT}"
ST=`date +%s`

for (( i=0; i < $IMAGE_COUNT; i++))
do
    if [ $i -eq 1 ] || ([ $i -ne 0 ] && [ $((i%PROCESS_CNT)) -eq 0 ])
    then
        wait
    fi
    echo -ne "Initializing process: $((i+1))/${IMAGE_COUNT}"\\r
    if [ $i -eq $((IMAGE_COUNT-1)) ]
    then
        echo "Initilization completed! ($((i+1))/${IMAGE_COUNT})"
        echo ""
    fi
    ID=`basename ${IMAGE_FILES[i]%.*}`
    ./parser/parser "${ID}" "`./process_image.sh ${IMAGE_FILES[i]}`" "${OUTPUT_FILE}" "${MODE}" &
    MODE="a"
done

wait
ET=`date +%s`
echo ""
echo "Done! (Took $((ET-ST)) seconds for captioning and parsing)"

./clean.sh
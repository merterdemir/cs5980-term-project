#!/bin/bash

#####Set Scheduler Configuration Directives#####
#Set the name of the job. This will be the first part of the error/output filename.
#$ -N text

#Set the current working directory as the location for the error and output files.
#(Will show up as .e and .o files)
#$ -cwd

#Send e-mail at beginning/end/suspension of job
##$ -m bes

#E-mail address to send to
##$ -M anna-keil@uiowa.edu
#####End Set Scheduler Configuration Directives#####


#####Resource Selection Directives#####
#See the HPC wiki for complete resource information: https://wiki.uiowa.edu/display/hpcdocs/Argon+Cluster
#Select the queue to run in
#$ -q CLAS-INSTR-GPU,UI-GPU

#Select the number of slots the job will use
#$ -pe smp 1

#Indicate that the job requires a GPU
#$ -l gpu=true

#Sets the number of requested GPUs to 1
#$ -l ngpus=1

#Indicate that the job requires a mid-memory (currently 256GB node)
#$ -l mem_256G=true

#Indicate the CPU architecture the job requires
##$ -l cpu_arch=broadwell

#Specify a data center for to run the job in
##$ -l datacenter=LC

#Specify the high speed network fabric required to run the job
##$ -l fabric=omnipath
#####End Resource Selection Directives#####


#####Begin Compute Work#####
#Print information from the job into the output file
/bin/echo Running on compute node: `hostname`.
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`

# module load tensorflow/1.7.0_python-3.6.4
module load tensorflow/1.7.0_cuda-9.0.176.1_python-3.6.4
module load python/3.6.4
pip install libhdf5-dev

python text_generator.py train

#Print the end date of the job before exiting
echo Now it is: `date`
#####End Compute Work#####


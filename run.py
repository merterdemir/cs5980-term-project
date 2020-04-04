#!/usr/bin/env python3

import os
import sys
import time
import shlex
import ntpath
import logging
import argparse
import subprocess
import multiprocessing
from generate_captions import _get_filename, _validate_parameters, _parameter_parser

def _get_image_names(path):
    VALID_EXTENTIONS = ('.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF')
    filenames = [f for f in os.listdir(path) if f.endswith(VALID_EXTENTIONS) and os.path.isfile(os.path.join(path, f))]
    return filenames

def _build_inference():
    logging.info("Building inference with Bazel.")
    bashCommand = "bazel build -c opt im2txt/run_inference"
    process = subprocess.Popen(shlex.split(bashCommand),
                               cwd=os.path.join(os.getcwd(), "im2txt"))
    output, error = process.communicate()
    if not error:
        logging.info("Build successful!")
        return True
    return False

def _clean_bazel():
    logging.info("Cleaning Bazel build.")
    bashCommand = "bazel clean"
    process = subprocess.Popen(shlex.split(bashCommand),
                               cwd=os.path.join(os.getcwd(), "im2txt"))
    output, error = process.communicate()
    if not error:
        process = subprocess.Popen(shlex.split("rm -rf /private/var/tmp/_bazel_*"),
                               cwd=os.path.join(os.getcwd(), "im2txt"))
        output, error = process.communicate()
        if not error:
            logging.info("Cleaned!")
            return True
    return False

def _create_captions(args):
    bashCommand  = "python3 generate_captions.py "
    bashCommand += "{} ".format(args.FILE)
    bashCommand += "-cp '{}' ".format(args.checkpoint_path)
    bashCommand += "-vf '{}' ".format(args.vocab_file)
    bashCommand += "-e '{}' ".format(args.export)
    bashCommand += "-et '{}' ".format(args.export_type)
    bashCommand += "-bs {}".format(args.beam_size)

    process = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if not error:
        logging.info("Success! ({})".format(_get_filename(args.FILE)))
    else:
        logging.error("Error! ({})".format(_get_filename(args.FILE)))

def start_captioning(args):
    try:
        logging.info("Starting to generate image captions.")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            p.map(_create_captions, args)
        logging.info("All processes are finalized!")
    except:
        logging.error("Error while initializing the processes.")

def main(args):
    if (_validate_parameters(args)):
        images  = _get_image_names(args.FILE)
        img_cnt = len(images)
        logging.info("Number of Images: {}".format(img_cnt))

        if not _build_inference():
            logging.error("Couldn't build bazel! Try again. Terminating!")
            sys.exit(-1)

        all_args = []
        for image in images:
            if image:
                new_args = argparse.Namespace(**vars(args))
                new_args.FILE = os.path.join(args.FILE, image)
                all_args.append(new_args)

        st = time.time()
        start_captioning(all_args)
        et = time.time()
        logging.info("Task completed in {} seconds!".format(et - st))
        if not _clean_bazel():
            logging.error("Couldn't clean the bazel build! Try again. Terminating!")
            sys.exit(-1)



if (__name__ == "__main__"):
    args = _parameter_parser()
    main(args)

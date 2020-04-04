#!/usr/bin/env python3

import os
import sys
import csv
import json
import shlex
import ntpath
import logging
import argparse
import subprocess
from collections import OrderedDict

def _get_filename(path):
    head, tail = ntpath.split(path)
    filename = tail or ntpath.basename(head)
    return filename.split(".")[0]

def _validate_parameters(args):
    # Initialize logging
    _id = _get_filename(args.FILE)
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s (Line %(lineno)d)")
    logging.info("Validating parameters...")
    if os.path.exists(args.FILE):
        if (os.getcwd() in args.FILE):
            logging.info("Images with id {} is ready to be captioned.".format(_id))
        else:
            args.FILE = os.path.join(os.getcwd(), args.FILE)
            logging.warning("Image path changed to full path. Image {} is ready to be captioned.".format(_id))
            logging.info("New path: {}".format(args.FILE))
    else:
        logging.error("IMAGE: " + args.FILE)
        logging.error("Given image with id {} doesn't exist! Terminating!".format(_id))
        return False

    if ((os.path.exists(args.checkpoint_path) or os.path.exists(args.checkpoint_path + ".data-00000-of-00001")) and 
       (os.path.exists(args.checkpoint_path + ".meta")) and (os.path.exists(args.checkpoint_path + ".index"))):
        logging.info("Checkpoint file exists.")
    else:
        logging.error("Checkpoint file doesn't exist! It should also have index and meta files. Terminating!")
        logging.error("Given path: {}".format(args.checkpoint_path))
        return False

    if os.path.exists(args.vocab_file):
        logging.info("Vocabulary file exists.")
    else:
        logging.error("Vocabulary file doesn't exist! Terminating!")
        return False

    if os.path.exists(args.export):
        logging.info("Export path exists.")
    else:
        logging.warning("Export folder doesn't exists. Creating the export folder.")
        os.mkdir(args.export)
    return True

def _parameter_parser():
    parser = argparse.ArgumentParser(description="Command Line Tool for generating image captions and parsing them to files.")

    parser.add_argument("FILE",
                        type=str,
                        help="Name/Path of the corresponding Image.")

    parser.add_argument("-cp", "--checkpoint_path",
                        type=str,
                        nargs='?',
                        default=os.path.join(os.getcwd(),
                                             'checkpoints',
                                             '5M',
                                             'model.ckpt-5000000'),
	                    help = "Path for the checkpoint of the pretrained model.")

    parser.add_argument("-vf", "--vocab_file",
                        type=str,
                        nargs='?',
                        default=os.path.join(os.getcwd(),
                                             'checkpoints',
                                             '5M',
                                             'word_counts.txt'),
	                    help = "Path for the vocabulary for the evaluation.")

    parser.add_argument("-e", "--export",
                        type=str,
                        nargs='?',
                        default=os.path.join(os.getcwd(),
                                             'captions'),
	                    help = "Export folder for the parsed data of the given image (As a path).")

    parser.add_argument("-et", "--export_type",
                        type=str,
                        nargs='?',
                        default='csv',
                        const='csv',
                        choices=['csv', 'json'],
	                    help = "Export file type of the result.")

    parser.add_argument("-bs", "--beam_size",
                        type=int,
                        nargs='?',
                        default=3,
	                    help = "Number of generated captions at the end.")

    return parser.parse_args()

def _export_parsed_data(_id, data, filename, filetype):
    try:
        logging.info("Moving to export operation for {}.".format(data['id']))
        with open(os.path.join(filename, _id + "." + filetype), "w") as f:
            if (filetype == "csv"):
                csvFile = csv.DictWriter(f, fieldnames=data.keys())
                csvFile.writeheader()
                csvFile.writerow(data)
            elif (filetype == "json"):
                f.write(json.dumps(data))
            return True
    except Exception as e:
        logging.error("Error while exporting {}: {}".format(data['id'], e))
        return False


def _parse_captions(_id, captionText, filename, filetype):
    try:
        logging.info("Parsing for {} is started.".format(_id))
        data         = OrderedDict({'id': _id})
        prob         = ""
        caption      = ""
        guessStrings = captionText.decode('utf-8').split("\n")

        for i in range(0, len(guessStrings)):
            if (guessStrings[i]):
                length  = len(guessStrings[i])
                prob    = float(guessStrings[i][length - 9:length - 1].strip())
                caption = guessStrings[i][5: length - 12]
                data["prediction{}".format(i)] = caption.strip()
                data["logprob{}".format(i)]    = prob

        logging.info("Captions successfully parsed.")
        return _export_parsed_data(_id, data, filename, filetype)
    except:
        logging.error("Error while parsing {}: {}.".format(_id), exc_info=True)
        return False

def _generate_captions(args):
    bashCommand  = "im2txt/bazel-bin/im2txt/run_inference "
    bashCommand += "--checkpoint_path='{}' ".format(args.checkpoint_path)
    bashCommand += "--vocab_file='{}' ".format(args.vocab_file)
    bashCommand += "--input_files='{}' ".format(args.FILE)
    bashCommand += "--beam_size={}".format(args.beam_size)

    _id = _get_filename(args.FILE)
    process = subprocess.Popen(shlex.split(bashCommand), stdout=subprocess.PIPE)
    output, error = process.communicate()

    if not error:
        logging.info("Successfully generated the captions for {}!".format(_id))
        logging.info("Moving to parsing stage.")
        if (_parse_captions(_id, output, args.export, args.export_type)):
            logging.info("Operations for {} are done!".format(_id))
        else:
            logging.error("Generating captions is unsuccessful for {}.".format(_id))
    else:
        logging.error("Couldn't generate the captions for {}".format(_id))

def main(args):
    if (_validate_parameters(args)):
        _generate_captions(args)

if (__name__ == "__main__"):
    args = _parameter_parser()
    main(args)

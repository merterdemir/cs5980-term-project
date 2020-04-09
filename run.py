#!/usr/bin/env python3

import os
import argparse
import ntpath
import logging
import time
import csv
import json

def _get_image_names(path):
    VALID_EXTENTIONS = ('.jpg', '.jpeg', '.png', '.gif', '.JPG', '.JPEG', '.PNG', '.GIF')
    filenames = [os.path.join(os.getcwd(),path,f) for f in os.listdir(path) if f.endswith(VALID_EXTENTIONS) and os.path.isfile(os.path.join(path, f))]
    return filenames

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

    if 
    return True

def _parameter_parser():
    parser = argparse.ArgumentParser(description="Command Line Tool for generating image captions and parsing them to files.")

    parser.add_argument("FILE",
                        type=str,
                        nargs = '?',
                        default=os.path.join(os.getcwd(),
                                             'images'),
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

    parser.add_argument("-m","--multiprocessing",
                        type=bool,
                        nargs='?',
                        default=True,
                        const=True,
                        choices=[True, False],
                        help = "Run on multiple threads.")

    parser.add_argument("-gpu","--GPU",
                        type=bool,
                        nargs='?',
                        default=False,
                        const=True,
                        choices=[True, False],
                        help = "Run on GPU.")

    return parser.parse_args()

def _export_data(data, filename, filetype, num_of_predictions):
    try:
        logging.info("Exporting captions to {}.".format(filetype))
        with open(os.path.join(filename + "." + filetype), "w") as f:
            if (filetype == 'csv'):
                fieldnames = ['id']
                for i in range(num_of_predictions):
                    fieldnames.append("prediction{}".format(i))
                    fieldnames.append("logprob{}".format(i))
                csvFile = csv.writer(f)
                csvFile.writerow(fieldnames)
                for d in data:
                    row = [os.path.splitext(os.path.basename(d[0]))[0]]
                    for cap in d[1]:
                        row.extend(cap)
                    csvFile.writerow(row)
            else:
                f.write(json.dumps(data))
    except Exception as e:
        logging.error("Error while exporting captions: "+str(e))
        return False

def main(args):
    from im2txt import run_inference
    filenames = _get_image_names(args.FILE)
    checkpoint_path = os.path.join(os.getcwd(), args.checkpoint_path)
    vocab_path = os.path.join(os.getcwd(), args.vocab_file)
    beam_size = args.beam_size
    logging.info("Generating Captions for {} images".format(len(filenames)))
    start = time.time()
    m = run_inference.generate_captions(checkpoint_path, vocab_path, filenames, beam_size, multiprocessing=args.multiprocessing, GPU=args.GPU)
    logging.info("Done!")
    logging.info("Generating captions for {} images took {:.2f} seconds".format(len(filenames),time.time()-start))
    _export_data(m, args.export, args.export_type, args.beam_size)
    # input()

if (__name__ == "__main__"):
    args = _parameter_parser()
    if _validate_parameters(args):
        main(args)

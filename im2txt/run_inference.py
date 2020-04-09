# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

import math

import os

import tensorflow as tf


# tf.logging.set_verbosity(tf.logging.INFO)
"""
Make Tensorflow less verbose
"""

try:
    # noinspection PyPackageRequirements
    import os
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
    # noinspection PyUnusedLocal
    def deprecated(date, instructions, warn_once=True):
        def deprecated_wrapper(func):
            return func
        return deprecated_wrapper

    from tensorflow.python.util import deprecation
    deprecation.deprecated = deprecated
except ImportError:
    pass

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

def generate_caption(filenames, checkpoint_path, vocab, beam_size):
    #import sys
    #sys.stdout = open(str(os.getpid())+".out",'a')
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               checkpoint_path)
    g.finalize()
    with tf.Session(graph = g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        result = []
        generator = caption_generator.CaptionGenerator(model, vocab, beam_size=beam_size)
        for filename in filenames:
            with tf.gfile.GFile(filename, "rb") as f:
                image = f.read()
            captions = generator.beam_search(sess, image)
            #print("Captions for image %s:" % os.path.basename(filename))
            cap = []
            for i, caption in enumerate(captions):
              # Ignore begin and end words.
              sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
              sentence = " ".join(sentence)
              cap.append((sentence,math.exp(caption.logprob)))
              #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
            result.append((filename,cap))
    return result

def generate_captions(checkpoint_path, vocab_file, filenames, beam_size, multiprocessing=True, GPU=False):
  if not GPU:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(vocab_file)

  tf.logging.info("Running caption generation on %d files",
                  len(filenames))

  if not multiprocessing:
      return generate_caption(filenames, checkpoint_path, vocab, beam_size)
  else:
      from functools import partial
      from multiprocessing import Pool,cpu_count

      part = partial(generate_caption, checkpoint_path=checkpoint_path, vocab=vocab, beam_size=beam_size)
      # Splits the list a into n equal sized parts
      def split(a, n):
          k, m = divmod(len(a), n)
          return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
      filenames = list(split(filenames, cpu_count()))
      with Pool(cpu_count()) as p:
          l = list(p.map(part, filenames))
      result = []
      for r in l:
          result.extend(r)
      return result

if __name__ == "__main__":
  tf.app.run()

import argparse
import cv2
import json
import os

import tensorflow as tf

from train import build_lstm_forward, build_overfeat_forward
from train import googlenet_load
from utils.train_utils import add_rectangles


def eval(H, ckpt_file, in_dir, out_dir, conf):
    """
    Re-constructs a TF based on the input checkpoint file (and the default googlenet graph) and applies it to the set
    of images in the input directory. If the output directory is supplied, annotated images are saved there, with detect
    boxes shown for any detect whose computed condifence exceeds 'conf'

    Note: the re
    """

    # load graph
    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in')
    # add the TF ops necessary for the reinspect algorihtm, based on the architecture defined in the hypes file
    if H['arch']['use_lstm']:
        pred_boxes, pred_logits, pred_confidences = build_lstm_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    else:
        pred_boxes, pred_logits, pred_confidences = build_overfeat_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, ckpt_file)

        # process the files
        save_output = False
        if out_dir != None and os.path.isdir(out_dir):
            save_output = True
        file_list = os.listdir(in_dir)
        filenames = next(os.walk(in_dir))[2]
        num_files = len(filenames)
        for i in range(1,num_files,1):
            f = filenames[i]
            img_raw = cv2.imread(os.path.join(in_dir,f))
            # Rudin images are 480 x 704, need them to be 480 x 640, so crop or resize ...
            img = img_raw[:,1:641,:]
            print 'Processing file:', f
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)

            new_img, rects = add_rectangles([img], np_pred_confidences, np_pred_boxes,
                                            H["arch"], use_stitching=True, rnn_len=H['arch']['rnn_len'], min_conf=conf)

            if True:
                cv2.imshow('Image with overlay',new_img)
                cv2.waitKey(1000)
                if save_output:
                    cv2.imwrite(os.path.join(out_dir,f), new_img)



def main():
    """
    Runs a tensorflow graph agains a set of images. (For now) Graphs are specified by a hyperparameter file and a
    checkpoint file.

    Command line arguments must include either:
        --img_dir, --graph_dir (this will load the hyperparmeters and checkpoint files from graph_dir)
        --img_dir, --ckpt, --hypes (this will use the specified hyperparmeters and checkpoint files)

    If --out_dir is specified, then the output imagery will be saved

    If --conf is specified, that value will be used as the minimum confidence (default = 0.8)
    """

    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', default=None, type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--hypes', default=None, type=str)
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--out_dir', default=None, type=str)
    parser.add_argument('--conf', default=0.8, type=float)
    args = parser.parse_args()

    # verify the input graph directory exists
    if not os.path.exists(args.graph_dir):
        print 'Input graph does not exist: ', args.in_graph
        return

    # verify the input image directory exists
    if not os.path.exists(args.img_dir):
        print 'Input directory does not exist: ', args.img_dir
        return

    if args.graph_dir is not None:
        hypes_file = os.path.join(args.graph_dir, 'hypes.json')

        # find the newest checkpoint file in the input directory
        all_files = os.listdir(args.graph_dir)
        # get a list of all files that start with save.ckpt and do not end in .meta
        all_ckpt_files = [f for f in all_files if f.startswith('save.') and not f.endswith('meta')]
        # verify there is at least one checkpoint file
        if len(all_ckpt_files) == 0:
            print 'No checkpoint files at: ', args.img_dir
            return

        ckpt_file = os.path.join(args.graph_dir, all_ckpt_files[-1])
        print '\thypes_file: ', hypes_file
        print '\tckpt_file: ', ckpt_file
    else:
        if args.hypes is None or args.ckpt is None:
            print 'Either --in_path or (--hypes and --ckpt) must be specified'
            return

        hypes_file = args.hypes
        ckpt_file = args.ckpt

    # verify the hypes file exists
    if not os.path.exists(hypes_file):
        print 'Hyperparameter file does not exist: ', hypes_file
        return

    if not os.path.exists(ckpt_file):
        print 'Checkpoint file does not exist: ', ckpt_file
        return

    # read in the hypes file
    with open(hypes_file, 'r') as f:
        H = json.load(f)


    eval(H, ckpt_file, args.img_dir, args.out_dir, args.conf)


if __name__ == '__main__':
    main()



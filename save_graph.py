# This script uses freeze_graph to construct a self-contained TF graph that can be used with the C++ API
import argparse
import json
import os

import tensorflow as tf

from train import build_lstm_forward, build_overfeat_forward
from train import googlenet_load

import freeze_graph

def save_graph(H, ckpt_file, output_graph_file):
    """
    Combines a checkpoint and a graph definition to create a self-contained output graph.
    """
    write_graph_to_tb = False
    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in')
    if H['arch']['use_lstm']:
        pred_boxes, pred_logits, pred_confidences = build_lstm_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    else:
        pred_boxes, pred_logits, pred_confidences = build_overfeat_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, ckpt_file)
        # all_vars = tf.all_variables()
        # for v in all_vars:
        #     print 'var: ', v.name

        # write out the graph def to be used by freeze_graph and then removed
        temp_path = "/Users/brucks/Desktop/"
        temp_graph_name = "temp_unneeded.pb"
        temp_graph_pathname = os.path.join(temp_path, temp_graph_name)
        tf.train.write_graph(sess.graph_def, temp_path, temp_graph_name, as_text=False)

        # call freeze_graph with the graph def and the checkpoint to save a combined graph that can be read into c++
        input_saver_def_path = ""
        input_binary = True
        input_checkpoint_path = ckpt_file
        if H['arch']['use_lstm']:
            output_node_names = "x_in,decoder/box_ip0,decoder/conf_ip0,decoder/box_ip,decoder/conf_ip1," \
                                "decoder/box_ip2,decoder/conf_ip2,decoder/box_ip3,decoder/conf_ip3," \
                                "decoder/box_ip4,decoder/conf_ip4"
        else:
            output_node_names = "x_in,pred_conf,pred_boxes"

        restore_op_name = "save/restore_all"
        filename_tensor_name = "save/Const:0"
        clear_devices = True
        initializer_nodes = ""
        success = freeze_graph.freeze_graph(temp_graph_pathname, input_saver_def_path,
                                  input_binary, input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_file,
                                  clear_devices, initializer_nodes)

        if write_graph_to_tb:
            output_path = os.path.dirname(output_graph_file)
            writer = tf.train.SummaryWriter(logdir=output_path) #,flush_secs=10)
            # add the graph def to the summary so it can be visualized
            writer.add_graph(sess.graph)

    # Print out a status message including the variable names saved in the graph
    if success >= 0:
        print 'Output graph saved to: ', output_graph_file
        vars = output_node_names.split(",")
        print '\tInput variable name: ', vars[0]
        print '\tOutput variable names: %s, %s' % (vars[1], vars[2])

        # Now remove the temporary graph that was created
        if os.path.exists(temp_graph_pathname):
            os.remove(temp_graph_pathname)
    else:
        print 'Error: graph not saved'


def main():
    """
    Combines a checkpoint and a graph definition to create a self-contained output graph.
    Command line arguments must include either:
        --out_graph, -in_dir
        --out_graph, --ckpt, --hypes
    """

    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default=None, type=str)
    parser.add_argument('--ckpt', default=None, type=str)
    parser.add_argument('--hypes', default=None, type=str)
    parser.add_argument('--out_graph', required=True, type=str)
    args = parser.parse_args()

    if args.in_dir is not None:
        hypes_file = os.path.join(args.in_dir, 'hypes.json')

        # find the newest checkpoint file in the input directory
        all_files = os.listdir(args.in_dir)
        # get a list of all files that start with save.ckpt and do not end in .meta
        all_ckpt_files = [f for f in all_files if f.startswith('save.') and not f.endswith('meta')]
        # verify there is at least one checkpoint file
        if len(all_ckpt_files) == 0:
            print 'No checkpoint files at: ', args.in_dir
            return

        ckpt_file = os.path.join(args.in_dir, all_ckpt_files[-1])
        print '\thypes_file: ', hypes_file
        print '\tckpt_file: ', ckpt_file

    else:
        if args.hypes is None or args.ckpt is None:
            print 'Either --in_path or (--hypes and --ckpt) must be specified'
            return

        hypes_file = args.hypes
        ckpt_file = args.ckpt

    # verify the ckpt_file, hypes_file and the path to args.out_graph all exist
    output_path = os.path.dirname(os.path.abspath(args.out_graph))
    if not os.path.exists(output_path):
        print 'Invalid path for output graph: ', output_path
        return

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
    # H = json.load(hypes_file)


    save_graph(H, ckpt_file, args.out_graph)

if __name__ == '__main__':
    main()


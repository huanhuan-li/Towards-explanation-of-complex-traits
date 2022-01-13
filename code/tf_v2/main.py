# coding=UTF-8

import os, json, time, random, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

import dataloader
from model import graphGAN
from utils import calcu_rsquare_distance
import warnings
warnings.filterwarnings("ignore")

#os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='/home/hli/Gene-Regulatory-Atlas/1_TFinfer/data_4GAN/MCA/imputation/all_per_celltype/links_from_magic_4genies_threshold002', help='path to input data')                                    
parser.add_argument('-f', type=str, default='Tcells_3cls_hard_forGAN_train.npy', help='input data file')
parser.add_argument('-fv', type=str, default='Tcells_3cls_hard_forGAN_val.npy', help='input data file for validation')
parser.add_argument('-ft', type=str, default='Tcells_3cls_hard_forGAN_test.npy', help='input data file for validation')
parser.add_argument('-spidx_file', type=str, default='layer_links.json', help='sparse index file name')
parser.add_argument('-denseshape_file', type=str, default='layer_dense_shape.json', help='sparse dense shape file name')
parser.add_argument('-save_dir', default='/home/hli/Gene-Regulatory-Atlas/1_TFinfer/infer_model/CE-ggan/out/')
parser.add_argument('-log_dir', default='/home/hli/Gene-Regulatory-Atlas/1_TFinfer/infer_model/CE-ggan/out/log/')
parser.add_argument('-checkpoint_dir', default='checkpoint/')
parser.add_argument('-sample_dir', default='sample/')
parser.add_argument('-load_model', default=None, type=str)
parser.add_argument('-batchsize', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('-numepoch', type=int, default=600001, metavar='N', help='epoch')
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-out', type=str, default=None, help='dir name')
parser.add_argument('-TEST_MODE', action='store_true', default=False, help='mode')
parser.add_argument('-TEST_OUT', type=int, default=None, help='global_step')
args = parser.parse_args()

'''save setting'''
if args.out == None:
    model_name = "test_run" + "_" + str(np.random.randint(1000))
else:
    model_name = args.out
log_dir = args.log_dir + model_name
checkpoint_dir = args.save_dir + args.checkpoint_dir + model_name
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

sample_dir = args.save_dir + args.sample_dir + model_name
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

'''load data'''
############## for train
input_param = {
    'path': os.path.join(args.d, args.f),
    'batchsize': args.batchsize,
    'input_data_type': 'float32',
    'shuffle': True
}
data_handler = dataloader.InputHandle(input_param)

############## for valid
if args.TEST_MODE:
    val_file = args.fv
else:
    val_file = args.fv
input_param_val = {
    'path': os.path.join(args.d, val_file),
    'batchsize': args.batchsize,
    'input_data_type': 'float32',
    'shuffle': False}
data_val = dataloader.InputHandle(input_param_val)

xdim = data_handler.xdim
vdim = 64
zdim = 128

'''build/load model'''
mdl = graphGAN(batchsize = args.batchsize,
            vdim = vdim,
            zdim = zdim,
            bdim = 32,
            xdim = xdim,
            path_to_sparse_indices = args.d,
            sparse_indices_filename = args.spidx_file,
            path_to_dense_shape = args.d,
            dense_shape_filename = args.denseshape_file,
            lr = 1e-4,
            beta1 = 0,
            beta2 = 0.999,)

'''train'''
n_critic = 10
def train():
    summary_writer = tf.summary.create_file_writer(log_dir)
    global_step=0
    for epoch in range(args.numepoch):
        for _ in range(n_critic):
            # train D
            batch_v = np.random.normal(0., 1., size=(args.batchsize, vdim)).astype(np.float32)
            batch_x = data_handler.samp_batch()
            mdl.train_D(batch_v=batch_v, batch_x=batch_x)
        # train G
        batch_v = np.random.normal(0., 1., size=(args.batchsize, vdim)).astype(np.float32)
        batch_x = data_handler.samp_batch()
        mdl.train_G(batch_v=batch_v, batch_x=batch_x)
        # edit on 2020/10/19, keep the batch (not use the full data_val.data in one run to avoid effect of batchnorm)
        if global_step % 1000 == 0:
            val_gen_pt = np.empty(shape=[0, mdl.tdim])
            val_gen_qt = np.empty(shape=[0, mdl.tdim])
            val_real_x = np.empty(shape=[0, xdim])
            val_gen_px = np.empty(shape=[0, xdim])
            val_recx_from_qt = np.empty(shape=[0, xdim])
            val_recx_from_qtb = np.empty(shape=[0, xdim])
            for i in range(data_val.N//args.batchsize):
                batch_v = np.random.normal(0., 1., size=(args.batchsize, vdim)).astype(np.float32)
                batch_x = data_val.samp_batch()
                pz = mdl.GzNet(batch_v, training=False)
                pt, pb = mdl.GtNet(pz, training=False)
                px = mdl.GxNet([pt, pb], training=False)
                qt, qb, _, _ = mdl.QNet(batch_x, training=False)
                rx = mdl.GxNet([qt, pb], training=False) #reconstructe x from qt and pb
                rx_ = mdl.GxNet([qt, qb], training=False) #reconstructe x from qt and q
                
                val_gen_pt = np.append(val_gen_pt, pt, axis=0)
                val_gen_qt = np.append(val_gen_qt, qt, axis=0)
                val_real_x = np.append(val_real_x, batch_x, axis=0)
                val_gen_px = np.append(val_gen_px, px, axis=0)
                val_recx_from_qt = np.append(val_recx_from_qt, rx, axis=0)
                val_recx_from_qtb = np.append(val_recx_from_qtb, rx_, axis=0)
            with summary_writer.as_default():
                tf.summary.scalar('Loss/D_zv', mdl.dzv_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/D_zt', mdl.dzt_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/D_zb', mdl.dzb_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/D_tx', mdl.dxtb_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/D_xx', mdl.dx_rect2x_loss_metric.result(), step=global_step)
                tf.summary.scalar('Loss/GP_zv', mdl.dzv_gp_metric.result(), step=global_step)
                tf.summary.scalar('Loss/GP_zt', mdl.dzt_gp_metric.result(), step=global_step)
                tf.summary.scalar('Loss/GP_zb', mdl.dzb_gp_metric.result(), step=global_step)
                tf.summary.scalar('Loss/GP_tx', mdl.dxtb_gp_metric.result(), step=global_step)
                tf.summary.scalar('Loss/GP_xx', mdl.dx_rect2x_gp_metric.result(), step=global_step)
                tf.summary.scalar('Similarity/val_GxX_realX_qsm', np.mean(calcu_rsquare_distance(val_real_x, val_gen_px)), step=global_step)
                tf.summary.scalar('Similarity/val_RecT2X_realX_qsm', np.mean(calcu_rsquare_distance(val_real_x, val_recx_from_qt)), step=global_step)
                tf.summary.scalar('Similarity/val_pT_qT_qsm', np.mean(calcu_rsquare_distance(val_gen_pt, val_gen_qt)), step=global_step)
            if global_step % 50000 == 0:
                np.save(os.path.join(sample_dir, 'val_px_step_{}.npy'.format(global_step)), val_gen_px)
                np.save(os.path.join(sample_dir, 'val_qt_step_{}.npy'.format(global_step)), val_gen_qt)
                np.save(os.path.join(sample_dir, 'val_qx_step_{}.npy'.format(global_step)), val_real_x)
                np.save(os.path.join(sample_dir, 'val_recx_from_qt_step_{}.npy'.format(global_step)), val_recx_from_qt)
                np.save(os.path.join(sample_dir, 'val_recx_from_qbt_step_{}.npy'.format(global_step)), val_recx_from_qtb)
        # save model
        if global_step % 100000 == 0:
            manager = tf.train.CheckpointManager(mdl.ckpt, str(checkpoint_dir + '/' + str(global_step)), max_to_keep=3)
            save_path = manager.save()
        global_step += 1
        
def test():
    manager = tf.train.CheckpointManager(mdl.ckpt, str(checkpoint_dir + '/' + str(args.TEST_OUT)), max_to_keep=3)
    mdl.ckpt.restore(manager.latest_checkpoint)
    
    ### generate samples
    batch_v = np.random.normal(0., 1., size=(data_val.N, vdim)).astype(np.float32)
    batch_x = data_val.data
    pz = mdl.GzNet(batch_v, training=False)
    pt, pb = mdl.GtNet(pz, training=False)
    px = mdl.GxNet([pt, pb], training=False)
    qt, qb, _, _ = mdl.QNet(batch_x, training=False)
    rx = mdl.GxNet([qt, pb], training=False) #reconstructe x from qt and pb
    rx_ = mdl.GxNet([qt, qb], training=False) #reconstructe x from qt and q
    var_ = [v for v in mdl.GxNet.trainable_variables if 'Gx.t2x' in v.name][0] 
    ### save
    np.save(os.path.join(sample_dir, 'val_px_step_{}.npy'.format(args.TEST_OUT)), px)
    np.save(os.path.join(sample_dir, 'val_qt_step_{}.npy'.format(args.TEST_OUT)), qt)
    np.save(os.path.join(sample_dir, 'val_qx_step_{}.npy'.format(args.TEST_OUT)), batch_x)
    np.save(os.path.join(sample_dir, 'val_recx_from_qt_step_{}.npy'.format(args.TEST_OUT)), rx)
    np.save(os.path.join(sample_dir, 'val_recx_from_qbt_step_{}.npy'.format(args.TEST_OUT)), rx_)
    np.save(os.path.join(sample_dir, 'weight_t2x_step_{}.npy'.format(args.TEST_OUT)), var_.numpy())

if args.TEST_MODE:
    test()
else:
    train()

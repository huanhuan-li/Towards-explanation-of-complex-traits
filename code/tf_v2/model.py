import os, json
import numpy as np
from functools import partial
import tensorflow as tf
from tensorflow.python.keras import layers, metrics, models
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()
from utils import BatchNorm, batchnorm, LeakyRelu, Relu, DenseLayer, SparseLinear

w_init = tf.keras.initializers.RandomNormal(stddev=0.02)
he_norm = tf.keras.initializers.he_normal()
he_unif = tf.keras.initializers.he_uniform()
var_scaling = tf.keras.initializers.VarianceScaling()

# gradient penalty
def gradient_penalty(discriminator, input_list, input_real_list, **kwargs):
    interpolations_list = []
    for input, input_real in zip(input_list, input_real_list):
        alpha = tf.random.uniform(shape=tf.shape(input), minval=0., maxval=1.)
        interpolations = alpha*input + (1.-alpha)*input_real
        interpolations_list.append(interpolations)
    with tf.GradientTape(persistent=True) as grad_tape:
        grad_tape.watch(interpolations_list)
        w = discriminator(interpolations_list)   
    grad_list = [grad_tape.gradient(w, [i])[0] for i in interpolations_list]
    gradients = tf.concat(grad_list, axis=1)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return gradient_penalty

class graphGAN():
    def __init__(self, batchsize,
                vdim, zdim, bdim, xdim,
                path_to_sparse_indices, sparse_indices_filename, path_to_dense_shape, dense_shape_filename, 
                lr, beta1, beta2,
                gp_lambda=10.):
        self.batchsize = batchsize
        self.vdim = vdim
        self.zdim = zdim
        self.bdim = bdim
        self.xdim = xdim
        self.path_to_sparse_indices = path_to_sparse_indices
        self.sparse_indices_filename = sparse_indices_filename
        with open(os.path.join(path_to_dense_shape, dense_shape_filename), 'r') as dense_shape_f:
            self.dense_shape = json.load(dense_shape_f)
        with open(os.path.join(self.path_to_sparse_indices, self.sparse_indices_filename), 'r') as sparse_indices_f:
            self.sparse_indices = json.load(sparse_indices_f)
        self.tdim = self.dense_shape['t2x'][1]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.gp_lambda = gp_lambda
        self.build()
        
    def build(self):
        # models
        self.GzNet = self.make_GzNet_model(init=he_unif)
        self.GtNet = self.make_GtNet_model(init=he_unif)
        self.GxNet = self.make_GxNet_model(init=he_unif)
        self.QNet = self.make_QNet_model(init=he_unif)
        self.DzvNet = self.make_DzvNet_model(init=he_unif)
        self.DztNet = self.make_DztNet_model(init=he_unif)
        self.DzbNet = self.make_DzbNet_model(init=he_unif)
        self.DxtbNet = self.make_DxtbNet_model(init=he_unif)
        self.Dx_rect2xNet = self.make_Dx_rect2xNet_model(init=he_unif)
        # optimizer
        self.G_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.beta1, beta_2=self.beta2)
        self.D_optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.beta1, beta_2=self.beta2)
        # loss metric
        self.dzv_loss_metric = tf.keras.metrics.Mean('dzv_loss', dtype=tf.float32)
        self.dzt_loss_metric = tf.keras.metrics.Mean('dzt_loss', dtype=tf.float32)
        self.dzb_loss_metric = tf.keras.metrics.Mean('dzb_loss', dtype=tf.float32)
        self.dxtb_loss_metric = tf.keras.metrics.Mean('dxtb_loss', dtype=tf.float32)
        self.dx_rect2x_loss_metric = tf.keras.metrics.Mean('dx_rect2x_loss', dtype=tf.float32)
        
        self.dzv_gp_metric = tf.keras.metrics.Mean('dzv_gp', dtype=tf.float32)
        self.dzt_gp_metric = tf.keras.metrics.Mean('dzt_gp', dtype=tf.float32)
        self.dzb_gp_metric = tf.keras.metrics.Mean('dzb_gp', dtype=tf.float32)
        self.dxtb_gp_metric = tf.keras.metrics.Mean('dxtb_gp', dtype=tf.float32)
        self.dx_rect2x_gp_metric = tf.keras.metrics.Mean('dx_rect2x_gp', dtype=tf.float32)
        # checkpoint
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        GzNet = self.GzNet,
                                        GtNet = self.GtNet,
                                        GxNet = self.GxNet,
                                        QNet = self.QNet,
                                        DzvNet = self.DzvNet,
                                        DztNet = self.DztNet,
                                        DzbNet = self.DzbNet,
                                        DxtbNet = self.DxtbNet,
                                        Dx_rect2xNet = self.Dx_rect2xNet)

    def make_GzNet_model(self, init):
        inputs = layers.Input(shape=(self.vdim,))
        #
        x = DenseLayer(64, init, 'Gz.0')(inputs)
        x = LeakyRelu()(x)
        #x = BatchNorm(name='Gz.0')(x)
        x = batchnorm(x, axis=[0])
        #
        x = DenseLayer(64, init, 'Gz.1')(x)
        x = LeakyRelu()(x)
        #x = BatchNorm(name='Gz.1')(x)
        x = batchnorm(x, axis=[0])
        #
        x = DenseLayer(self.zdim, init, 'Gz.out')(x)
        x = LeakyRelu()(x)
        return models.Model(inputs=inputs, outputs=x, name="generate_z")
        
    def make_GtNet_model(self, init):
        inputs = layers.Input(shape=(self.zdim,))
        #
        x = DenseLayer(512, init, 'Gt.0')(inputs)
        x = LeakyRelu()(x)
        #x = BatchNorm(name='Gt.0')(x)
        x = batchnorm(x, axis=[0])
        #
        x = DenseLayer(512, init, 'Gt.1')(x)
        x = LeakyRelu()(x)
        #x = BatchNorm(name='Gt.1')(x)
        x = batchnorm(x, axis=[0])
        #
        t = DenseLayer(self.tdim, init, 'Gt.out.t')(x)
        t = Relu()(t)
        #
        b = DenseLayer(self.bdim, init, 'Gt.out.b')(x)
        b = LeakyRelu()(b)
        return models.Model(inputs=inputs, outputs=[t, b], name="generate_t_and_b")
        
    def make_GxNet_model(self, init):
        inputA = layers.Input(shape=(self.tdim,))
        inputB = layers.Input(shape=(self.bdim,))
        ### the first branch
        xA = SparseLinear(indices = self.sparse_indices['x2t'],
                            n_feat = self.tdim,
                            units = self.xdim,
                            init = init,
                            w_name = 'Gx.t2x')(inputA)
        ### the second branch
        xB = DenseLayer(512, init, 'Gx.b2x.0')(inputB)
        xB = LeakyRelu()(xB)
        #xB = BatchNorm(name='Gx.b2x.0')(xB)
        xB = batchnorm(xB, axis=[0])
        xB = DenseLayer(self.xdim, init, 'Gx.b2x.1')(xB)
        xB = LeakyRelu()(xB)
        ### combine the output of the two branches
        x = layers.Add()([xA, xB])
        x = Relu()(x)
        return models.Model(inputs=[inputA, inputB], outputs=x)
        
    def make_QNet_model(self, init):
        inputs = layers.Input(shape=(self.xdim,))
        #
        qt = SparseLinear(indices = self.sparse_indices['t2x'],
                            n_feat = self.xdim,
                            units = self.tdim,
                            init = init,
                            w_name = 'Gv.x2t')(inputs)
        qt = Relu()(qt)
        #
        qb = DenseLayer(512, init, 'Gv.x2b.0')(inputs)
        qb = LeakyRelu()(qb)
        #qb = BatchNorm(name='Gv.x2b.0')(qb)
        qb = batchnorm(qb, axis=[0])
        qb = DenseLayer(32, init, 'Gv.x2b.out')(qb)
        qb = LeakyRelu()(qb)
        #
        qz = layers.Concatenate(axis=1)([qt, qb])
        qz = DenseLayer(512, init, 'Gv.t2z.0')(qz)
        qz = LeakyRelu()(qz)
        #qz = BatchNorm(name='Gv.t2z.0')(qz)
        qz = batchnorm(qz, axis=[0])
        qz = DenseLayer(512, init, 'Gv.t2z.1')(qz)
        qz = LeakyRelu()(qz)
        #qz = BatchNorm(name='Gv.t2z.1')(qz)
        qz = batchnorm(qz, axis=[0])
        qz = DenseLayer(self.zdim, init, 'Gv.t2z.out')(qz)
        qz = LeakyRelu()(qz)
        #
        qv = DenseLayer(64, init, 'Gv.z2v.0')(qz)
        qv = LeakyRelu()(qv)
        #qv = BatchNorm(name='Gv.z2v.0')(qv)
        qv = batchnorm(qv, axis=[0])
        qv = DenseLayer(64, init, 'Gv.z2v.1')(qv)
        qv = LeakyRelu()(qv)
        #qv = BatchNorm(name='Gv.z2v.1')(qv)
        qv = batchnorm(qv, axis=[0])
        qv = DenseLayer(2*self.vdim, init, 'Gv.z2v.out')(qv)
        return models.Model(inputs=inputs, outputs=[qt, qb, qz, qv])
        
    def make_DzvNet_model(self, init):
        inputA = layers.Input(shape=(self.zdim,))
        inputB = layers.Input(shape=(self.vdim,))
        #
        h_z = DenseLayer(64, init, 'Dzv.z.0')(inputA)
        h_z = LeakyRelu()(h_z)
        h_z = layers.Dropout(0.2)(h_z)
        h_z = DenseLayer(64, init, 'Dzv.z.1')(h_z)
        h_z = LeakyRelu()(h_z)
        h_z = layers.Dropout(0.2)(h_z)
        #
        h_v = DenseLayer(32, init, 'Dzv.v.0')(inputB)
        h_v = LeakyRelu()(h_v)
        h_v = layers.Dropout(0.2)(h_v)
        h_v = DenseLayer(32, init, 'Dzv.v.1')(h_v)
        h_v = LeakyRelu()(h_v)
        h_v = layers.Dropout(0.2)(h_v)
        #
        h_zv = layers.Concatenate(axis=1)([h_z, h_v])
        h_zv = DenseLayer(64, init, 'Dzv.zv.0')(h_zv)
        h_zv = LeakyRelu()(h_zv)
        h_zv = layers.Dropout(0.2)(h_zv)
        h_zv = DenseLayer(1, init, 'Dzv.zv.out')(h_zv)
        return models.Model(inputs=[inputA, inputB], outputs=h_zv)
        
    def make_DztNet_model(self, init):
        inputA = layers.Input(shape=(self.zdim,))
        inputB = layers.Input(shape=(self.tdim,))
        #
        h_z = DenseLayer(64, init, 'Dzt.z.0')(inputA)
        h_z = LeakyRelu()(h_z)
        h_z = layers.Dropout(0.2)(h_z)
        h_z = DenseLayer(64, init, 'Dzt.z.1')(h_z)
        h_z = LeakyRelu()(h_z)
        h_z = layers.Dropout(0.2)(h_z)
        #
        h_t = DenseLayer(256, init, 'Dzt.t.0')(inputB)
        h_t = LeakyRelu()(h_t)
        h_t = layers.Dropout(0.2)(h_t)
        h_t = DenseLayer(256, init, 'Dzt.t.1')(h_t)
        h_t = LeakyRelu()(h_t)
        h_t = layers.Dropout(0.2)(h_t)
        h_t = DenseLayer(256, init, 'Dzt.t.2')(h_t)
        h_t = LeakyRelu()(h_t)
        h_t = layers.Dropout(0.2)(h_t)
        #
        h_zt = layers.Concatenate(axis=1)([h_z, h_t])
        h_zt = DenseLayer(256, init, 'Dzt.zt.0')(h_zt)
        h_zt = LeakyRelu()(h_zt)
        h_zt = layers.Dropout(0.2)(h_zt)
        h_zt = DenseLayer(1, init, 'Dzt.zt.out')(h_zt)
        return models.Model(inputs=[inputA, inputB], outputs=h_zt)
        
    def make_DzbNet_model(self, init):
        inputA = layers.Input(shape=(self.zdim,))
        inputB = layers.Input(shape=(self.bdim,))
        #
        h_z = DenseLayer(64, init, 'Dzb.z.0')(inputA)
        h_z = LeakyRelu()(h_z)
        h_z = layers.Dropout(0.2)(h_z)
        h_z = DenseLayer(64, init, 'Dzb.z.1')(h_z)
        h_z = LeakyRelu()(h_z)
        h_z = layers.Dropout(0.2)(h_z)
        #
        h_b = DenseLayer(32, init, 'Dzb.b.0')(inputB)
        h_b = LeakyRelu()(h_b)
        h_b = layers.Dropout(0.2)(h_b)
        h_b = DenseLayer(32, init, 'Dzb.b.1')(h_b)
        h_b = LeakyRelu()(h_b)
        h_b = layers.Dropout(0.2)(h_b)
        #
        h_zb = layers.Concatenate(axis=1)([h_z, h_b])
        h_zb = DenseLayer(64, init, 'Dzb.zb.0')(h_zb)
        h_zb = LeakyRelu()(h_zb)
        h_zb = layers.Dropout(0.2)(h_zb)
        h_zb = DenseLayer(1, init, 'Dzb.zb.out')(h_zb)
        return models.Model(inputs=[inputA, inputB], outputs=h_zb)
    
    def make_DxtbNet_model(self, init):
        inputA = layers.Input(shape=(self.xdim,))
        inputB = layers.Input(shape=(self.tdim,))
        inputC = layers.Input(shape=(self.bdim,))
        #
        h_x = DenseLayer(512, init, 'Dxtb.x.0')(inputA)
        h_x = LeakyRelu()(h_x)
        h_x = layers.Dropout(0.2)(h_x)
        h_x = DenseLayer(512, init, 'Dxtb.x.1')(h_x)
        h_x = LeakyRelu()(h_x)
        h_x = layers.Dropout(0.2)(h_x)
        h_x = DenseLayer(512, init, 'Dxtb.x.2')(h_x)
        h_x = LeakyRelu()(h_x)
        h_x = layers.Dropout(0.2)(h_x)
        #
        h_t = DenseLayer(256, init, 'Dxtb.t.0')(inputB)
        h_t = LeakyRelu()(h_t)
        h_t = layers.Dropout(0.2)(h_t)
        h_t = DenseLayer(256, init, 'Dxtb.t.1')(h_t)
        h_t = LeakyRelu()(h_t)
        h_t = layers.Dropout(0.2)(h_t)
        h_t = DenseLayer(256, init, 'Dxtb.t.2')(h_t)
        h_t = LeakyRelu()(h_t)
        h_t = layers.Dropout(0.2)(h_t)
        #
        h_b = DenseLayer(32, init, 'Dxtb.b.0')(inputC)
        h_b = LeakyRelu()(h_b)
        h_b = layers.Dropout(0.2)(h_b)
        h_b = DenseLayer(32, init, 'Dxtb.b.1')(h_b)
        h_b = LeakyRelu()(h_b)
        h_b = layers.Dropout(0.2)(h_b)
        #
        h_xtb = layers.Concatenate(axis=1)([h_x, h_t, h_b])
        h_xtb = DenseLayer(64, init, 'Dxtb.xtb.0')(h_xtb)
        h_xtb = LeakyRelu()(h_xtb)
        h_xtb = layers.Dropout(0.2)(h_xtb)
        h_xtb = DenseLayer(1, init, 'Dxtb.xtb.out')(h_xtb)
        return models.Model(inputs=[inputA, inputB, inputC], outputs=h_xtb)
        
    def make_Dx_rect2xNet_model(self, init):
        inputA = layers.Input(shape=(self.xdim,))
        inputB = layers.Input(shape=(self.xdim,))
        #
        h_x = DenseLayer(512, init, 'Dx_rect2x.x.0')(inputA)
        h_x = LeakyRelu()(h_x)
        h_x = layers.Dropout(0.2)(h_x)
        h_x = DenseLayer(512, init, 'Dx_rect2x.x.1')(h_x)
        h_x = LeakyRelu()(h_x)
        h_x = layers.Dropout(0.2)(h_x)
        h_x = DenseLayer(512, init, 'Dx_rect2x.x.2')(h_x)
        h_x = LeakyRelu()(h_x)
        h_x = layers.Dropout(0.2)(h_x)
        #
        h_x_ = DenseLayer(512, init, 'Dx_rect2x.rect2x.0')(inputB)
        h_x_ = LeakyRelu()(h_x_)
        h_x_ = layers.Dropout(0.2)(h_x_)
        h_x_ = DenseLayer(512, init, 'Dx_rect2x.rect2x.1')(h_x_)
        h_x_ = LeakyRelu()(h_x_)
        h_x_ = layers.Dropout(0.2)(h_x_)
        h_x_ = DenseLayer(512, init, 'Dx_rect2x.rect2x.2')(h_x_)
        h_x_ = LeakyRelu()(h_x_)
        h_x_ = layers.Dropout(0.2)(h_x_)
        #
        h_xx = layers.Concatenate(axis=1)([h_x, h_x_])
        h_xx = DenseLayer(512, init, 'Dx_rect2x.xrect2x.0')(h_xx)
        h_xx = LeakyRelu()(h_xx)
        h_xx = layers.Dropout(0.2)(h_xx)
        h_xx = DenseLayer(1, init, 'Dx_rect2x.xrect2x.out')(h_xx)
        return models.Model(inputs=[inputA, inputB], outputs=h_xx)
        
    def sample_from_latent_distribution(self, z_mean, z_logvar):
        return tf.add(z_mean, tf.exp(z_logvar / 2) * tf.random.normal(tf.shape(z_mean), 0, 1), name="sampled_latent_variable")
    
    @tf.function
    def train_D(self, batch_v, batch_x):
        with tf.GradientTape(persistent=True) as grad_tape:
            '''generate samples'''
            pz = self.GzNet(batch_v, training=True)
            pt, pb = self.GtNet(pz, training=True)
            px = self.GxNet([pt, pb], training=True)
            qt, qb, qz, qv_params = self.QNet(batch_x, training=True)
            qv_mu = qv_params[:, :self.vdim]
            qv_logvar = qv_params[:, self.vdim:]
            qv = self.sample_from_latent_distribution(qv_mu, qv_logvar)
            rx = self.GxNet([qt, pb], training=True) #reconstructe x from qt and pb
            
            D_P_zv = self.DzvNet([pz, batch_v], training=True)
            D_Q_zv = self.DzvNet([qz, qv], training=True)
            D_P_zt = self.DztNet([pz, pt], training=True)
            D_Q_zt = self.DztNet([qz, qt], training=True)
            D_P_zb = self.DzbNet([pz, pb], training=True)
            D_Q_zb = self.DzbNet([qz, qb], training=True)
            D_P_xtb = self.DxtbNet([px, pt, pb], training=True)
            D_Q_xtb = self.DxtbNet([batch_x, qt, qb], training=True)
            D_realx = self.Dx_rect2xNet([batch_x, batch_x], training=True)
            D_rect2x = self.Dx_rect2xNet([batch_x, rx], training=True)
            '''losses'''
            dzv_loss = tf.reduce_mean(D_P_zv) - tf.reduce_mean(D_Q_zv)
            dzv_gp = gradient_penalty(partial(self.DzvNet, training=True), [pz, batch_v], [qz, qv])
            dzv_loss += self.gp_lambda * dzv_gp
            
            dzt_loss = tf.reduce_mean(D_P_zt) - tf.reduce_mean(D_Q_zt)
            dzt_gp = gradient_penalty(partial(self.DztNet, training=True), [pz, pt], [qz, qt])
            dzt_loss += self.gp_lambda * dzt_gp
            
            dzb_loss = tf.reduce_mean(D_P_zb) - tf.reduce_mean(D_Q_zb)
            dzb_gp = gradient_penalty(partial(self.DzbNet, training=True), [pz, pb], [qz, qb])
            dzb_loss += self.gp_lambda * dzb_gp
            
            dxtb_loss = tf.reduce_mean(D_P_xtb) - tf.reduce_mean(D_Q_xtb)
            dxtb_gp = gradient_penalty(partial(self.DxtbNet, training=True), [px, pt, pb], [batch_x, qt, qb])
            dxtb_loss += self.gp_lambda * dxtb_gp
            
            dxx_loss = tf.reduce_mean(D_rect2x) - tf.reduce_mean(D_realx)
            dxx_gp = gradient_penalty(partial(self.Dx_rect2xNet, training=True), [batch_x, rx], [batch_x, batch_x])
            dxx_loss += self.gp_lambda * dxx_gp
            
        for loss, var in zip([dzv_loss, dzt_loss, dzb_loss, dxtb_loss, dxx_loss], 
                             [self.DzvNet.trainable_variables, self.DztNet.trainable_variables, self.DzbNet.trainable_variables, self.DxtbNet.trainable_variables, self.Dx_rect2xNet.trainable_variables]):
            D_grads = grad_tape.gradient(loss, var)
            self.D_optimizer.apply_gradients(zip(D_grads, var))
        
        self.dzv_gp_metric(self.gp_lambda * dzv_gp)
        self.dzt_gp_metric(self.gp_lambda * dzt_gp)
        self.dzb_gp_metric(self.gp_lambda * dzb_gp)
        self.dxtb_gp_metric(self.gp_lambda * dxtb_gp)
        self.dx_rect2x_gp_metric(self.gp_lambda * dxx_gp)
        
    @tf.function
    def train_G(self, batch_v, batch_x):
        with tf.GradientTape(persistent=True) as grad_tape:
            '''generate samples'''
            pz = self.GzNet(batch_v, training=True)
            pt, pb = self.GtNet(pz, training=True)
            px = self.GxNet([pt, pb], training=True)
            qt, qb, qz, qv_params = self.QNet(batch_x, training=True)
            qv_mu = qv_params[:, :self.vdim]
            qv_logvar = qv_params[:, self.vdim:]
            qv = self.sample_from_latent_distribution(qv_mu, qv_logvar)
            rx = self.GxNet([qt, pb], training=True) #reconstructe x from qt and pb
            
            D_P_zv = self.DzvNet([pz, batch_v], training=True)
            D_Q_zv = self.DzvNet([qz, qv], training=True)
            D_P_zt = self.DztNet([pz, pt], training=True)
            D_Q_zt = self.DztNet([qz, qt], training=True)
            D_P_zb = self.DzbNet([pz, pb], training=True)
            D_Q_zb = self.DzbNet([qz, qb], training=True)
            D_P_xtb = self.DxtbNet([px, pt, pb], training=True)
            D_Q_xtb = self.DxtbNet([batch_x, qt, qb], training=True)
            D_realx = self.Dx_rect2xNet([batch_x, batch_x], training=True)
            D_rect2x = self.Dx_rect2xNet([batch_x, rx], training=True)
            '''losses'''
            g_loss = tf.reduce_mean(D_Q_zv) - tf.reduce_mean(D_P_zv) + \
                     tf.reduce_mean(D_Q_zt) - tf.reduce_mean(D_P_zt) + \
                     tf.reduce_mean(D_Q_zb) - tf.reduce_mean(D_P_zb) + \
                     tf.reduce_mean(D_Q_xtb) - tf.reduce_mean(D_P_xtb) + \
                     tf.reduce_mean(D_realx) - tf.reduce_mean(D_rect2x)
            
        self.Gx_vars = self.GzNet.trainable_variables + self.GtNet.trainable_variables + self.GxNet.trainable_variables
        self.Gv_vars = self.QNet.trainable_variables
        G_grads = grad_tape.gradient(g_loss, self.Gx_vars+self.Gv_vars)
        self.G_optimizer.apply_gradients(zip(G_grads, self.Gx_vars+self.Gv_vars))

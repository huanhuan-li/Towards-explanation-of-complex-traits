import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from scipy.stats import percentileofscore

class BatchNorm(layers.Layer):
    '''https://github.com/drewszurko/tensorflow-WGAN-GP/blob/master/ops.py
       # why axis=-1? 
       # When we compute a BatchNormalization along an axis, we preserve the dimensions of the array, 
       # and we normalize with respect to the mean and standard deviation over every other axis. 
       # So in your 2D example BatchNormalization with axis=1 is subtracting the mean for axis=0 '''
    def __init__(self, epsilon=1e-3, axis=-1, name=None):
        super(BatchNorm, self).__init__()
        self.batch_norm = layers.BatchNormalization(axis=axis,
                                                    epsilon=epsilon,
                                                    trainable=False, # important to repeat the result of tf_v1 code
                                                    name=name)

    def call(self, inputs, **kwargs):
        return self.batch_norm(inputs)
        
def batchnorm(inputs, axis, scale=None, offset=None, variance_epsilon=0.001, name=None):
    mean, var = tf.nn.moments(inputs, axis, keepdims=True)
    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, variance_epsilon, name=name)
    return result

class Relu(layers.Layer):
    def __init__(self):
        super(Relu, self).__init__()
        self.relu = layers.ReLU()

    def call(self, inputs, **kwargs):
        return self.relu(inputs)

class LeakyRelu(layers.Layer):
    def __init__(self, alpha=0.2):
        super(LeakyRelu, self).__init__()
        self.leaky_relu = layers.LeakyReLU(alpha=alpha)

    def call(self, inputs, **kwargs):
        return self.leaky_relu(inputs)
        
class DenseLayer(layers.Layer):
    def __init__(self, hidden_n, init, name):
        super(DenseLayer, self).__init__()
        self.fc_op = layers.Dense(hidden_n,
                                  kernel_initializer=init,
                                  bias_initializer=tf.keras.initializers.Constant(value=0.0),
                                  name=name)

    def call(self, inputs):
        x = self.fc_op(inputs)
        return x
        
class SparseLinear(layers.Layer):
    '''https://stackoverflow.com/questions/62153815/tensorflow-2-how-to-create-custom-layer-gradient-with-a-predifined-sparse-weigh'''
    def __init__(self, indices, n_feat, units, init, w_name):
        super(SparseLinear, self).__init__()
        self.n_feat = n_feat
        self.units = units
        self.indices = indices
        self.w_name = w_name
        self.w = self.add_weight(name=self.w_name,
                                 shape=(len(self.indices),),
                                 initializer=init,
                                 trainable=True)
    def call(self, x):
        kernel = tf.SparseTensor(self.indices, self.w, [self.n_feat, self.units])
        return tf.sparse.sparse_dense_matmul(x, kernel)
        
def calcu_rsquare_distance(ref_, samp_):
    '''
    TODO:
        Rsquare of quantile values genes in samp_ and ref_;
        Input:
            ndarray
        Output:
            list
    '''
    '''
    regr = LinearRegression()
    result = []
    for gene_idx in range(ref_.shape[1]):
        ref = ref_[:, gene_idx]
        samp = samp_[:, gene_idx]
        sample = np.round(samp, 3)
        
        samp_pct_x = []
        samp_pct_y = []
        processed_pct_x = {}
        processed_pct_y = {}
        
        for i,s in enumerate(list(sorted(sample))):
            if s in processed_pct_x.keys():
                samp_pct_x.append(processed_pct_x[s])
                samp_pct_y.append(processed_pct_y[s])                
            else:
                # theoretical quantiles
                processed_pct_x[s]=percentileofscore(ref, s)
                # sample quantiles
                processed_pct_y[s]=percentileofscore(samp, s)
                samp_pct_x.append(processed_pct_x[s])
                samp_pct_y.append(processed_pct_y[s]) 
 
        # estimated linear regression model
        samp_pct_x = np.array(samp_pct_x)
        samp_pct_y = np.array(samp_pct_y)
        model_x = samp_pct_x.reshape(len(samp_pct_x), 1)
        model_y = samp_pct_y.reshape(len(samp_pct_y), 1)
        regr.fit(model_x, model_y)
        r2 = regr.score(model_x, model_y)
        result.append(r2)
    '''
    result = []
    for gene_idx in range(ref_.shape[1]):
        ref = ref_[:, gene_idx]
        samp = samp_[:, gene_idx]
        
        sample = [0.02] + [i for i in np.arange(0.05, 1.0, 0.05)]
        
        samp_pct_x = []
        samp_pct_y = []
        
        for i,s in enumerate(sample):
            # theoretical quantiles
            samp_pct_x.append(percentileofscore(ref, s))
            # sample quantiles
            samp_pct_y.append(percentileofscore(samp, s)) 
        
        # estimated quantile distance
        r = np.linalg.norm(np.subtract(np.asarray(samp_pct_x)/100, np.asarray(samp_pct_y)/100), ord=2)
        result.append(r)
    return result

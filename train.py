import logging
import pickle
from typing import Optional, Mapping, Any

import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import haiku.initializers as hki
import einops
import functools as ft

import numpy as np
import optax
import pandas as pd

from statistics import mean
from sklearn.preprocessing import StandardScaler
import math
from sklearn.preprocessing import MinMaxScaler

def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=1e-6, name=name)(x)

class Time2Vec(hk.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        self.k = kernel_size

    def __call__(self, inputs):
        ii1 = inputs.shape[1]
        init = hki.RandomUniform(0, 1.0 / ii1)
        bias = hk.get_parameter('wb', shape=(ii1,), init=init) * inputs + hk.get_parameter('bb', shape=(ii1,), init=init)
        wa = hk.get_parameter('wa', shape=(1, ii1, self.k), init=init)
        ba = hk.get_parameter('ba', shape=(1, ii1, self.k), init=init)
        dp = jnp.dot(inputs, wa) + ba
        weights = jnp.sin(dp)

        ret = jnp.concatenate([jnp.expand_dims(bias, axis=-1), weights], -1)
        ret = einops.rearrange(ret, "t b c -> t (b c)")
        return ret


class AttentionBlock(hk.Module):
    def __init__(self, num_heads, head_size, ff_dim=None, dropout=0.0):
        super().__init__()
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0
        out_features = inputs.shape[-1]

        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.head_size, w_init_scale=1.0)(inputs, inputs, inputs)
        x = hk.BatchNorm(False, False, decay_rate=0.9, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = layer_norm(x)

        x = hk.Conv1D(output_channels=self.ff_dim, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.9, eps=1e-6)(x, is_training)
        x = jnn.gelu(x)
        x = hk.Conv1D(output_channels=out_features, kernel_shape=1, padding="same")(x)
        x = hk.BatchNorm(False, False, decay_rate=0.9, eps=1e-6)(x, is_training)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnn.gelu(x)

        return layer_norm(x + inputs)


class TimeDistributed(hk.Module):
    def __init__(self, module, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.module = module

    def __call__(self, x):
        module = self.module
        if len(x.shape) <= 2:
            return module(x)

        x_reshape = einops.rearrange(x, "b c h -> (b c) h")

        y = module(x_reshape)

        return jnp.where(self.batch_first, jnp.reshape(y, newshape=(x.shape[0], -1, y.shape[-1])), jnp.reshape(y, newshape=(-1, x.shape[1], y.shape[-1])))


class TransformerThunk(hk.Module):
    def __init__(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0):
        super().__init__()
        self.time2vec_dim = time2vec_dim
        if ff_dim is None:
            self.ff_dim = head_size
        else:
            self.ff_dim = ff_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers

    def __call__(self, inputs, is_training=True):
        dropout = self.dropout if is_training else 0.0
        bs = inputs.shape[0]
        time2vec = Time2Vec(kernel_size=self.time2vec_dim)
        time_embedding = TimeDistributed(time2vec)(inputs)

        x = jnp.concatenate([inputs, time_embedding], axis=-1)
        
        w_init = hki.VarianceScaling(1.0, mode='fan_in', distribution='truncated_normal')
        for i in range(self.num_layers):
            x = AttentionBlock(num_heads=self.num_heads, head_size=self.head_size, ff_dim=self.ff_dim, dropout=self.dropout)(x, is_training)
        x = jnp.dot(x, hk.get_parameter('wfinal', shape=(x.shape[2], 256), init=w_init)) + hk.get_parameter('biasfinal', shape=(256,), init=hki.Constant(1e-6))
        x = jnn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), dropout, x)
        x = jnp.dot(x, hk.get_parameter('wwfinal', shape=(256, 65), init=w_init)) + hk.get_parameter('bbiasfinal', shape=(65,), init=hki.Constant(1e-8))
        
        return x[:, -36:, :]

def build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, ff_dim=None, dropout=0.5):
    def forward_fn(x: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        tr = TransformerThunk(time2vec_dim, num_heads, head_size, ff_dim, num_layers, dropout)
        return tr(x, is_training)

    return forward_fn
        
     
@ft.partial(jax.jit, static_argnums=(0, 6))
def lm_loss_fn(forward_fn, params, state, rng, x, y, is_training: bool = True) -> jnp.ndarray:
    y_pred, state = forward_fn(params, state, rng, x, is_training)
    return jnp.mean(jnp.abs(y - y_pred)), state


class GradientUpdater:
    def __init__(self, net_init, loss_fn, optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, master_rng, x):
        out_rng, init_rng = jax.random.split(master_rng)
        params, state = self._net_init(init_rng, x)
        opt_state = self._opt.init(params)
        return jnp.array(0), out_rng, params, state, opt_state

    def update(self, num_steps, rng, params, state, opt_state, x:jnp.ndarray, y: jnp.ndarray):
        rng, new_rng = jax.random.split(rng)

        (loss, state), grads = jax.value_and_grad(self._loss_fn, has_aux=True)(params, state, rng, x, y)

        #loss = jax.lax.pmean(loss, axis_name='j')

        grads = jax.lax.pmean(grads, axis_name='j')

        updates, opt_state = self._opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        metrics = {
            'step': num_steps,
            'loss': loss,
        }

        return num_steps + 1, new_rng, params, state, opt_state, metrics

def get_generator(x, y, rng_key, batch_size, num_devices):
    def batch_generator():
        n = x.shape[0]
        key = rng_key
        kk = batch_size // num_devices
        while True:
            key, k1 = jax.random.split(key)
            perm = jax.random.choice(k1, n, shape=(batch_size,))
            
            yield x[perm, :, :].reshape(num_devices, kk, *x.shape[1:]), y[perm].reshape(num_devices, kk, *y.shape[1:])
    return batch_generator()

def replicate(t, num_devices):
    return jax.tree_map(lambda x: jnp.array([x] * num_devices), t)


def load(filename='./data/train.csv', filename1='./data/test.csv', filename2 = './data/sample_submission.csv'):
    train_df = pd.read_csv(filename)
    test_df = pd.read_csv(filename1)
    submission = pd.read_csv(filename2)

    data = train_df.pivot_table(index = 'time', columns = ['direction','x','y'], fill_value = 0, values = 'congestion')

    datatime = pd.to_datetime(data.index)

    sc = MinMaxScaler()
    datanorm = sc.fit_transform(data)
    tm = (datatime.hour.to_numpy()*3 + datatime.minute.to_numpy()/20)
    datanorm= np.append(datanorm,datatime.weekday.to_numpy()[:,np.newaxis]/6, axis = 1) 
    datanorm= np.append(datanorm, tm[:,np.newaxis]/71, axis = 1) 
    Ndays = 2
    lenseq = int(3*24*Ndays-36)
    Xtrain = np.empty((1,lenseq, datanorm.shape[1]))
    Ytrain = np.empty((1,3*12, datanorm.shape[1]-2))

    T = 0
    for t in range(0, datanorm.shape[0]-lenseq-12*12, 6) :
        if t//1000 > T:
            T = t//1000
        Xtemp = np.reshape(datanorm[t:t+lenseq,:],(1,lenseq, datanorm.shape[1]))
        Ytemp = np.reshape(datanorm[t+lenseq:t+lenseq+3*12,:datanorm.shape[1]-2],(1,3*12, datanorm.shape[1]-2))
            
        Xtrain = np.append(Xtrain,Xtemp, axis = 0)
        Ytrain = np.append(Ytrain,Ytemp, axis = 0)
        #Ytrain.append(Ytemp)

    Xtrain = Xtrain[1:,:,:]   
    Ytrain = Ytrain[1:,:,:]  
    Xtrain[np.isnan(Xtrain)] = 0
    Xtest = np.reshape(datanorm[-lenseq:,:],(1,lenseq, datanorm.shape[1]))

    dftest = test_df.sort_values(['time','direction','x','y'])

    return jnp.array(Xtrain), jnp.array(Ytrain), jnp.array(Xtest), dftest, sc


def main():
    max_steps = 200
    num_heads = 4
    head_size = 128
    num_layers = 2
    dropout_rate = 0.3
    grad_clip_value = 1.0
    learning_rate = 0.001
    time2vec_dim = 36
    batch_size = 512
    
    num_devices = jax.local_device_count()

    print("Num devices :::: ", num_devices)

    x, y, x_test, dftest, sc = load()

    print("Examples :::: ", x.shape)
    print("Example y shape", y.shape)
    print("Testing Examples :::: ", x_test.shape)

    rng1, rng = jr.split(jax.random.PRNGKey(18))
    train_dataset = get_generator(x, y, rng1, batch_size, num_devices)

    forward_fn = build_forward_fn(num_layers, time2vec_dim, num_heads, head_size, dropout=dropout_rate)

    forward_fn = hk.transform_with_state(forward_fn)

    forward_apply = forward_fn.apply
    loss_fn = ft.partial(lm_loss_fn, forward_apply)

    optimizer = optax.chain(
        optax.adaptive_grad_clip(grad_clip_value),
        #optax.sgd(learning_rate=learning_rate, momentum=0.95, nesterov=True),
        optax.radam(learning_rate=learning_rate)
    )

    updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

    logging.info('Initializing parameters...')
    rng1, rng = jr.split(rng)
    a = next(train_dataset)
    w, z = a
    num_steps, rng, params, state, opt_state = updater.init(rng1, w[0, :, :, :])

    params_multi_device = params
    opt_state_multi_device = opt_state
    num_steps_replicated = num_steps
    rng_replicated = rng
    state_multi_device = state

    fn_update = jax.pmap(updater.update, axis_name='j', in_axes=(None, None, None, None, None, 0, 0), out_axes=(None, None, None, None, None, 0))

    logging.info('Starting train loop ++++++++...')
    for i, (w, z) in zip(range(max_steps), train_dataset):
        if (i + 1) % 1 == 0:
            logging.info(f'Step {i} computing forward-backward pass')
        num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, metrics = \
            fn_update(num_steps_replicated, rng_replicated, params_multi_device, state_multi_device, opt_state_multi_device, w, z)

        if (i + 1) % 1 == 0:
            logging.info(f'At step {i} the loss is {metrics}')
    
    # Test part of the model
    forward_apply = jax.jit(forward_apply, static_argnames=['is_training'])
    params_reduced = params_multi_device # Reduce parameters for single device
    state_reduced = state_multi_device
    rng = rng_replicated

    Ypred, _ = forward_apply(params_reduced, state_reduced, rng, x_test, is_training=False)
    Ypred = np.array(Ypred)
    Ypred = sc.inverse_transform(Ypred[0, :, :])

    Ypred2 = np.ravel(Ypred[:36,:])

    dftest.insert(1, 'congestion', Ypred2)
    dftest = dftest.drop(columns=['direction','x','y','time'])
    
    submission = dftest.sort_values(['row_id'])
    submission['congestion'] = round(submission['congestion'] )
    submission.to_csv('./data/result_submissions.csv', index=False)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()

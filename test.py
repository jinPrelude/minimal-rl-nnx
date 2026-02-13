import tensorflow_datasets as tfds
import tensorflow as tf

train_steps = 1200
eval_every = 200
batch_size = 32

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds = tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
    lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 255,
        'label': sample['label']
    }
)
test_ds = test_ds.map(
    lambda sample: {
        'image': tf.cast(sample['image'], tf.float32) / 255,
        'label': sample['label']
    }
)

train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

# for step, batch in tqdm(enumerate(train_ds.as_numpy_iterator())):

############ model ############
from tqdm import tqdm
from functools import partial
from flax import nnx
import optax

class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3,3), rngs=rngs)
        self.dropout1 = nnx.Dropout(0.025)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3,3), rngs=rngs)
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.025)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

        self.avg_pool = partial(nnx.avg_pool, window_shape=(2,2), strides=(2,2))

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        x = self.avg_pool(self.dropout1(nnx.relu(self.conv1(x)), rngs=rngs))
        x = self.avg_pool(nnx.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.dropout2(self.linear1(x), rngs=rngs))
        return self.linear2(x)

def loss_fn(model: nnx.Module, rngs: nnx.Rngs, batch):
    logits = model(batch['image'], rngs=rngs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
    return loss, logits

def optimize(model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, rngs: nnx.Rngs, batch):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, rngs, batch)
    metrics.update(logits=logits, labels=batch['label'], loss=loss)
    optimizer.update(model, grads)

import pdb

def main():
    rngs = nnx.Rngs(0)
    model = CNN(rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=5e-4), wrt=nnx.Param)
    metrics = nnx.MultiMetric(accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average('loss'))


    for iter, batch in tqdm(enumerate(train_ds.as_numpy_iterator())):
        optimize(model, optimizer, metrics, rngs, batch)
        pdb.set_trace()



if __name__=="__main__":
    main()
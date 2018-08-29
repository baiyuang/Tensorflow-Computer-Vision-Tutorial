"""
A simple implementation of DCGAN that works on MNIST dataset.
[DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
Learn more, visit my tutorial site: [莫烦Python](https://morvanzhou.github.io)
Dependencies:
tensorflow=1.8.0
numpy=1.14.3
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# training parameters
BATCH_SIZE = 100
LEARNING_RATE = 0.001
N_EPOCHS = 4

# load MNIST
f = np.load('../mnist.npz')
train_x = f['x_train'][:, :, :, None]


def generator(x):
    # Input: random vector of size 100, Output: image to be inputted in discriminator
    with tf.variable_scope('generator'):
        net = tf.layers.conv2d_transpose(
            inputs=x,                                   # in: [batch, 1, 1, 100]
            filters=512,
            kernel_size=[4, 4],
            strides=(1, 1),
            activation=tf.nn.leaky_relu,
            padding='valid')                            # -> [batch, 4, 4, 512]
        net = tf.layers.conv2d_transpose(
            net, 256, [4, 4], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same')  # -> [batch, 8, 8, 256]
        net = tf.layers.conv2d_transpose(
            net, 128, [4, 4], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same')  # -> [batch, 16, 16, 128]
        net = tf.layers.conv2d_transpose(
            net, 1, [4, 4], strides=(2, 2), activation=tf.nn.tanh, padding='same')          # -> [batch, 32, 32, 1]
        return net


def discriminator(x, reuse=False):                                # reuse when predict fake image
    # Input: real data / output of generator, Output: a probability of the input image that is similar to real data
    with tf.variable_scope('discriminator', reuse=reuse):
        net = tf.layers.conv2d(
            inputs=x,                                                               # in: [batch, 32, 32, 1]
            filters=128,
            kernel_size=[4, 4],
            strides=(2, 2),
            activation=tf.nn.leaky_relu,
            padding='same')                                                         # -> [batch, 16, 16, 128]
        net = tf.layers.conv2d(
            net, 256, [4, 4], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same')      # -> [batch, 8, 8, 256]
        net = tf.layers.conv2d(
            net, 512, [4, 4], strides=(2, 2), activation=tf.nn.leaky_relu, padding='same')      # -> [batch, 4, 4, 512]
        logits = tf.layers.conv2d(
            net, 1, [4, 4], strides=(1, 1), padding='valid')                                    # -> [batch, 1, 1, 1]
        return logits, tf.nn.sigmoid(logits)    # get the probability


def show_result(num_epoch, ax, imgs, save=False):
    for k in range(5*5):
        i, j = k // 5, k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(imgs[k], (32, 32)), cmap='gray')
    plt.show();plt.pause(0.1)
    if save: plt.savefig("../results/dcgan_ep%i.png" % num_epoch)


# input placeholders
x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))     # [batch, width, height, channel]
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))

# generator
G_z = generator(z)

# discriminator
D_real_logits, D_real = discriminator(x)
D_fake_logits, D_fake = discriminator(G_z, reuse=True)

# losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([BATCH_SIZE, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([BATCH_SIZE, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([BATCH_SIZE, 1, 1, 1])))

# trainable variables
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizers
D_train = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.5).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator"))
G_train = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.5).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator"))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# resize and normalization real input dataset as mentioned in DCGAN paper.
train_set = tf.image.resize_images(train_x, [32, 32]).eval()
train_set = train_set / 255. * 2 - 1.  # scale to [-1 ~ 1] (It's the range of tanh activation)
n_batches = train_x.shape[0] // BATCH_SIZE

# for plotting
fixed_z = np.random.uniform(-1, 1, (25, 1, 1, 100))
fig, ax = plt.subplots(5, 5, figsize=(5, 5))
for i, j in itertools.product(range(5), range(5)):
    ax[i, j].get_xaxis().set_visible(False)
    ax[i, j].get_yaxis().set_visible(False)

# training
plt.ion()
for epoch in range(N_EPOCHS):
    show_result(epoch, ax, imgs=sess.run(G_z, {z: fixed_z}), save=True)
    for b in range(n_batches):
        # step of discriminator
        x_ = train_set[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        z_ = np.random.uniform(-1, 1, (BATCH_SIZE, 1, 1, 100))
        loss_d_, _ = sess.run([D_loss, D_train], {x: x_, z: z_})

        # step of generator
        z_ = np.random.uniform(-1, 1, (BATCH_SIZE, 1, 1, 100))
        loss_g_, _ = sess.run([G_loss, G_train], {z: z_, x: x_})
    print('[%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), N_EPOCHS, loss_d_, loss_g_))
plt.ioff()

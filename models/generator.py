from models.rnn import RNN

import numpy as np
import tensorflow as tf


def PPO_train( y_true, y_pred_old, y_pred, reward,num_emb,epsilon):
    y_pred = tf.reshape(y_pred, [-1, num_emb])
    y_pred_old = tf.reshape(y_pred_old, [-1, num_emb])
    ratio = y_pred / (y_pred_old + 1e-5)
    real_ratio = tf.reduce_sum(tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_emb, 1.0, 0.0) * ratio)
    surr = real_ratio * tf.reshape(reward * 64*51, [-1])  # surr是有比例的收获
    XXX = tf.minimum(surr,
                     tf.clip_by_value(real_ratio, 1. - epsilon, 1. + epsilon) * tf.reshape(reward* 64*51, [-1]))
    '''求均值的损失函数   上面的reward也需要调整·'''
    g_loss_sum = -tf.reduce_sum(XXX)  # 正常的不求期望 这两者没有本质区别只有嵌套了一个64的batchsize
    g_loss_mean = -tf.reduce_mean(XXX)  # 这就是期望
    '''有了损失函数还要有   重要采样的实现  在一个batch中多次训练  再把参数给old'''
    return g_loss_mean

class Generator(RNN):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length,epsilon, learning_rate=0.01,):
        super(Generator, self).__init__(num_emb, batch_size, emb_dim, hidden_dim, sequence_length, learning_rate)
        self.epsilon = epsilon
        self.generator_model = tf.keras.models.Sequential(self.generator_model.layers)
        self.oldG = tf.keras.models.Sequential(self.generator_model.layers)
        self.generator_optimizer = self._create_optimizer(
            learning_rate,
            clipnorm=self.grad_clip
        )
        self.generator_model.compile(
            optimizer=self.generator_optimizer,
            loss="sparse_categorical_crossentropy",
            sample_weight_mode="temporal")

    def pretrain(self, dataset, num_epochs, num_steps):
        #不要数据的最后一列然后再第一列前面加上一列0
        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", 0), x)).repeat(
            num_epochs)#给真实数据添加的参数为
        pretrain_loss = self.generator_model.fit(ds, verbose=1, epochs=num_epochs, steps_per_epoch=num_steps)
        self.oldG.set_weights(self.generator_model.get_weights())
        #print("Pretrain generator loss: ", pretrain_loss)
        return pretrain_loss

    def train_step(self, x, rewards):
        a = np.pad(x[:, 0:-1], ([0, 0], [1, 0]), "constant", constant_values=0)
        print(x)
        #print(a)
        print('#############################3333333333333333#################################')
        b = rewards * self.batch_size * self.sequence_length
        #print(b)
        print('##############################################################')
        #a 是输入  X是输出
        y = self.generator_model(a)
        #print(y)
        train_loss = self.generator_model.train_on_batch(a,
            x,
            sample_weight=b
        )
        print("Generator Loss: ", train_loss)
        return train_loss
    def train_step_PPO(self, x, rewards,times):#更新Generator参数####利用计算得到的reward来计算#############这里的损失函数不能用交叉熵
        #要使用和Wgan-GP相同的损失函数
        a = np.pad(x[:, 0:-1], ([0, 0], [1, 0]), "constant", constant_values=0)
        a_4 = np.split(a, times, axis=0)
        x_4 = np.split(x, times, axis=0)
        reward_4 = np.split(rewards, times, axis=0)
        #把a裁剪成N等分，rewards也要等分  把等分的数据依次投入训练   切分方向和乘积是否对
        train_loss = []
        for i in range(times):
            with tf.GradientTape() as total_tape:
                y_pred = self.generator_model(a_4[i])
                y_pred_old = self.oldG(a_4[i])
                w = PPO_train( x_4[i], y_pred_old, y_pred, reward_4[i],self.num_emb,self.epsilon)
                train_loss.append(w)
                print(w)
                gradients = total_tape.gradient(w, self.generator_model.trainable_variables)
                self.generator_optimizer.apply_gradients(zip(gradients, self.generator_model.trainable_variables))
        self.oldG.set_weights(self.generator_model.get_weights())  # 获得参数并把参数付给roll——out
        return sum(train_loss)
    def _create_optimizer(self, *args, **kwargs):
        return tf.keras.optimizers.Adam(*args, **kwargs)

    def save(self, filename):
        self.generator_model.save_weights(filename, save_format="h5")

    def load(self, filename):
        self.generator_model.load_weights(filename)

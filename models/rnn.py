import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Embedding, Dense, LSTM, GRU
import numpy as np





class RNN(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, learning_rate=0.01):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token_vec = tf.constant([0] * self.batch_size, dtype=tf.int32)
        self.learning_rate = learning_rate
        self.grad_clip = 5.0

        def lstm_model():#原始的生成器结构  要是改这个  那么蒙特卡洛树搜索的过程也要更改。
            return tf.keras.models.Sequential([
                Input((self.sequence_length,), dtype=tf.int32),
                Embedding(self.num_emb, self.emb_dim, embeddings_initializer=tf.random_normal_initializer(stddev=0.1)),
                LSTM(self.hidden_dim, kernel_initializer=tf.random_normal_initializer(stddev=0.1),recurrent_initializer=tf.random_normal_initializer(stddev=0.1), return_sequences=True),
                #GRU(self.hidden_dim, kernel_initializer=tf.random_normal_initializer(stddev=0.1),recurrent_initializer=tf.random_normal_initializer(stddev=0.1), return_sequences=True),
                Dense(self.num_emb, kernel_initializer=tf.random_normal_initializer(stddev=0.1), activation="softmax")
            ])

        self.generator_model = lstm_model()
        self.g_optimizer = self._create_optimizer(learning_rate, clipnorm=self.grad_clip)
        if self.g_optimizer is not None:
            self.generator_model.compile(
                optimizer=self.g_optimizer,
                loss="sparse_categorical_crossentropy")
        else:
            self.generator_model.compile(
                loss="sparse_categorical_crossentropy")
        self.g_embeddings = self.generator_model.trainable_weights[0]

    def target_loss(self, dataset):
        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", 0), x))
        loss = self.generator_model.evaluate(ds, verbose=1)
        return loss

    @tf.function
    def generate_one_batch(self):
        h0 = c0 = tf.zeros([self.batch_size, self.hidden_dim])
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)
#一个一个时间步长来计算生成，不是一次全部生成出来,这里实现了蒙特卡洛树搜索
        def _g_recurrence(i, x_t, h_tm1, gen_x):
            o_t, h_t = self.generator_model.layers[1].cell(x_t, h_tm1, training=False)  # layers[1]: LSTM
            o_t = self.generator_model.layers[2](o_t)  # layers[2]: Dense
            log_prob = tf.math.log(o_t)
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_x

        _, _, _, gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token_vec), [h0, c0], gen_x))

        gen_x = gen_x.stack()
        outputs = tf.transpose(gen_x, perm=[1, 0])
        print(outputs)
        return outputs

    def generate_samples(self, num_batches, output_file):
        with open(output_file, 'w') as fout:
            for _ in range(num_batches):
                generated_samples = self.generate_one_batch().numpy()
                for poem in generated_samples:
                    print(' '.join([str(x) for x in poem]), file=fout)

    def _create_optimizer(self, *args, **kwargs):
        return None


class gpt(object):
    def __init__(self, num_emb, batch_size, emb_dim, hidden_dim, sequence_length, learning_rate=0.01):
        self.num_emb = num_emb
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.start_token_vec = tf.constant([0] * self.batch_size, dtype=tf.int32)
        self.learning_rate = learning_rate
        self.grad_clip = 5.0

        inputs = layers.Input(shape=(self.sequence_length,), dtype=tf.int32)
        embedding_layer = Embedding(self.num_emb, self.emb_dim)
        x = embedding_layer(inputs)


        # 定义位置编码层

        position_embedding_layer = Embedding(input_dim=self.sequence_length, output_dim=self.emb_dim)
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        positions = position_embedding_layer(positions)
        x = x + positions


        # 定义多头自注意力层和前馈神经网络层

        for i in range(2):
            # 多头自注意力层
            attention_output = layers.MultiHeadAttention(num_heads=2, key_dim=self.emb_dim)(x, x)
            # 残差连接和层归一化
            x = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
            # 前馈神经网络层
            ffn_output = layers.Dense(self.emb_dim, activation='relu')(x)
            ffn_output = layers.Dense(self.emb_dim)(ffn_output)
            # 残差连接和层归一化
            x = layers.LayerNormalization(epsilon=1e-6,name='MHA'+i)(ffn_output + x)

        outputs = layers.Dense(num_emb)(x)

        def gpt_model():  # gptMODEL
            return keras.Model(inputs=inputs, outputs=outputs)

        self.generator_model = gpt_model()
        self.g_optimizer = self._create_optimizer(learning_rate, clipnorm=self.grad_clip)
        if self.g_optimizer is not None:
            self.generator_model.compile(
                optimizer=self.g_optimizer,
                loss="sparse_categorical_crossentropy")
        else:
            self.generator_model.compile(
                loss="sparse_categorical_crossentropy")
        self.g_embeddings = self.generator_model.trainable_weights[0]

    def target_loss(self, dataset):
        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", 0), x))
        loss = self.generator_model.evaluate(ds, verbose=1)
        return loss

    @tf.function
    def generate_one_batch(self):
        h0 = c0 = tf.zeros([self.batch_size, self.hidden_dim])
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)

        # 一个一个时间步长来计算生成，不是一次全部生成出来,这里实现了蒙特卡洛树搜索
        def _g_recurrence(i, x_t, h_tm1, gen_x):
            o_t, h_t = self.generator_model.layers[1].cell(x_t, h_tm1, training=False)  # layers[1]: LSTM
            o_t = self.generator_model.layers[2](o_t)  # layers[2]: Dense
            log_prob = tf.math.log(o_t)
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, gen_x

        _, _, _, gen_x = tf.while_loop(
            cond=lambda i, _1, _2, _3: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token_vec), [h0, c0], gen_x))

        gen_x = gen_x.stack()
        outputs = tf.transpose(gen_x, perm=[1, 0])
        print(outputs)
        return outputs

    def generate_one_batch2(self):
        start_token_vec = tf.zeros([self.batch_size, self.hidden_dim])
        gen_x = tf.TensorArray(dtype=tf.int32, size=self.sequence_length,
                               dynamic_size=False, infer_shape=True)

        # 一个一个时间步长来计算生成，不是一次全部生成出来,这里实现了蒙特卡洛树搜索
        def _g_recurrence(i, x_t, gen_x):
            o_t = self.generator_model(x_t, training=False)  # GPT 模型
            o_t = o_t[:, -1, :]  # 取最后一个时刻的输出
            log_prob = tf.math.log(o_t)
            next_token = tf.cast(tf.reshape(tf.random.categorical(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.concat([x_t, tf.expand_dims(next_token, 1)], axis=1)  # 将新 token 添加到序列中
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, gen_x

        _, _, gen_x = tf.while_loop(
            cond=lambda i, _1, _2: i < self.sequence_length,
            body=_g_recurrence,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.expand_dims(start_token_vec, 1), gen_x))

        gen_x = gen_x.stack()
        outputs = tf.transpose(gen_x, perm=[1, 0])
        print(outputs)
        return outputs

    def generate_samples(self, num_batches, output_file):
        with open(output_file, 'w') as fout:
            for _ in range(num_batches):
                generated_samples = self.generate_one_batch().numpy()
                for poem in generated_samples:
                    print(' '.join([str(x) for x in poem]), file=fout)

    def _create_optimizer(self, *args, **kwargs):
        return None
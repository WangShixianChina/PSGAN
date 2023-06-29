import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 定义模型参数
vocab_size = 20
maxlen = 20
embed_dim = 256
num_heads = 2
ff_dim = 256

# 定义输入层
inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)

# 定义词嵌入层
embedding_layer = layers.Embedding(vocab_size, embed_dim)
x = embedding_layer(inputs)

# 定义位置编码层
position_embedding_layer = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
positions = tf.range(start=0, limit=maxlen, delta=1)
positions = position_embedding_layer(positions)
x = x + positions

# 定义多头自注意力层和前馈神经网络层
for i in range(2):
    # 多头自注意力层
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    # 残差连接和层归一化
    x = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
    # 前馈神经网络层
    ffn_output = layers.Dense(ff_dim, activation='relu')(x)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    # 残差连接和层归一化
    x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + x)

# 定义输出层
outputs = layers.Dense(vocab_size)(x)

# 构建模型
model = keras.Model(inputs=inputs, outputs=outputs)

# 编译模型
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# 打印模型结构
model.summary()



def generate_text(model, sequence_length, tokenizer):
    # 初始化生成序列
    generated_tokens = []
    # 逐步生成序列
    for i in range(sequence_length):
        # 准备模型输入数据
        input_data = tf.expand_dims(generated_tokens, 0)
        # 使用模型预测下一个 token 的概率分布
        predictions = model.predict(input_data)
        # 从概率分布中采样下一个 token
        next_token_probs = predictions[0, -1, :]
        next_token = tf.random.categorical(tf.math.log([next_token_probs]), num_samples=1).numpy()[0][0]
        # 将预测结果添加到序列中
        generated_tokens.append(next_token)
    # 将 token 序列转换为文本
    generated_text = tokenizer.sequences_to_texts([generated_tokens])[0]
    return generated_text

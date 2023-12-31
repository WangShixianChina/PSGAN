import tensorflow as tf
def get_padding_mask(seq):
    with tf.name_scope("Padding_Mask"):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions to add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def attention_mask(size):
    """
    if size is 4 then it returns below matrix
       [[0., 1., 1., 1.],
        [0., 0., 1., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 0.]]

    """
    with tf.name_scope("attention_mask"):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


def create_masks(inp):
    with tf.name_scope("att_masking"):
        att_mask = attention_mask(tf.shape(inp)[1])
        padding_mask = get_padding_mask(inp)
        mask = tf.maximum(padding_mask, att_mask)

        return mask

inputs = tf.constant([1,2,3,4,5,6])

v = tf.cast(tf.not_equal(inputs, 0),tf.int32)
print(v)
inputs *= v
print(inputs)

decoder_layers = [v for _ in range(20)]
print(decoder_layers)
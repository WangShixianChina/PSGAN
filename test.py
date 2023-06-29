import tensorflow as tf
from models.generator import Generator
from settings import *
from preprocessing import load_tokenizer

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)


def load_generator():

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, epsilon=0.1)
    generator.load(generator_file)
    return generator

def decode_to_sentences(sentences):
    return tokenizer.sequences_to_texts(sentences)


if __name__ == "__main__":
    generator = load_generator()
    #tokenizer = load_tokenizer()
    '''生成样本规模'''
    seq_number = 132000
    negative_file = 'dataset/general_Y.txt'
    generator.generate_samples(seq_number // BATCH_SIZE, negative_file)
    #sentences = tokenizer.sequences_to_texts(generated_sentences)
    #print(*sentences, sep='\n')

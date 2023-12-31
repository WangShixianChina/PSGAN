import tensorflow as tf

import os
from utils.dataloader import generator_dataloader, discriminator_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from models.rollout import ROLLOUT
from settings import *

if __name__ == "__main__":
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow **IS** using the GPU")
    else:
        print("TensorFlow **IS NOT** using the GPU")
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,epsilon=0.1)
    discriminator = Discriminator(sequence_length=SEQ_LENGTH, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  dropout_keep_prob=dis_dropout_keep_prob,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    gen_dataset = generator_dataloader(positive_file, BATCH_SIZE)

    if not os.path.exists("pretrained_models"):
        os.makedirs("pretrained_models")

    if not os.path.exists(pretrained_generator_file):
        print('Start pre-training generator')
        generator.pretrain(gen_dataset, PRE_EPOCH_NUM, generated_num // BATCH_SIZE)#generated_num // BATCH_SIZE运转次数
        generator.save(pretrained_generator_file)
        print('Finished pre-training generator...')
    else:
        generator.load(pretrained_generator_file)

    if not os.path.exists(pretrained_discriminator_file):
        print('Start pre-training discriminator...')
        for _ in range(50):
            print("Dataset", _)
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
        discriminator.save(pretrained_discriminator_file)
        print('Finished pre-training discriminator...')
    else:
        discriminator.load(pretrained_discriminator_file)

    rollout = ROLLOUT(generator, 0.8)

    print('#########################################################################')
    print('Start Adversarial Training...')

    for epoch in range(EPOCH_NUM):
        print("Generator", epoch)
        for it in range(1):
            samples = generator.generate_one_batch()
            rewards = rollout.get_reward(samples, 16, discriminator)
           # generator.train_step(samples, rewards)#sample是样本生成的样本  rewards
            generator.train_step_PPO(samples, rewards, 4)
        rollout.update_params()

        print("Discriminator", epoch)
        for _ in range(2):
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    generator.save(generator_file)
    discriminator.save(discriminator_file)

    generator.generate_samples(generated_num // BATCH_SIZE, generated_file)

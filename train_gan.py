from working_simple_gan import *
image_size = (32,32)
batch_size = 128
latent_dim = 500
gan, discriminator, generator = get_gan()
epochs = 300
train_gen, val_gen = get_pcam_generators(r'C:\Users\justi\PycharmProjects\pythonProject\train+val',
                                         image_size, batch_size, batch_size)
train_gan(discriminator, generator, gan, epochs, batch_size, latent_dim, train_gen)

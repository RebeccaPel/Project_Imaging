import sys
import keras
from keras.utils.vis_utils import plot_model
sys.path.append("tools")
from custom_layers import MinibatchDiscrimination
from GAN import get_generator_histopathology
init = keras.initializers.get("glorot_uniform")
model = keras.models.load_model(r"C:\Users\justi\Documents\Project_Imaging\Main "
                                r"project\models\gan_discriminator_epoch_Upsampling_190.h5",
                                custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination,
                                                'GlorotUniform': init})
generator = get_generator_histopathology(500)
plot_model(generator, to_file='model_plot3.png', show_shapes=True, show_layer_names=True)
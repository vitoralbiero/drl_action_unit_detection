from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import SGD
from fexp.layers import RoiLayer
from keras_vggface.vggface import VGGFace
from keras import backend as K


INPUT_NAME = 'images'
AU_OCCURRENCE_OUTPUT_NAME = 'au_occurrence'
N_AUS_OCCURRENCE = 10
REGION_HEIGHT = 3
REGION_WIDTH = 3


def compile(model, writer):
    lr = 0.001
    momentum = 0.9
    decay = 1e-6

    writer.log_hyper_param('learning_rate_initial', lr)
    writer.log_hyper_param('momentum', momentum)
    writer.log_hyper_param('decay', decay)

    optmizer = SGD(lr=lr, decay=decay, momentum=momentum)

    loss = ['binary_crossentropy']

    loss_weights = [1.0]

    model.compile(optimizer=optmizer,
                  loss=loss,
                  loss_weights=loss_weights)


def model(shape, weights='vggface'):
    landmarks = Input(shape=(20, 2), name='landmarks')
    vgg_model = VGGFace(include_top=False,
                        input_shape=(224, 224, 3),
                        weights=weights)
    last_layer = vgg_model.get_layer('pool5').output

    # DYNAMIC REGIONS OF INTEREST
    roi_layer = RoiLayer()
    roi_layer.split(layer=last_layer,
                    landmarks=landmarks,
                    original_size=shape,
                    region_height=REGION_HEIGHT,
                    region_width=REGION_WIDTH)

    # regions operations
    roi_layer.add(_region_operation)

    # regions concat
    fcs_concatenated = roi_layer.concatenate_fully_connected()
    drop1 = Dropout(0.5)(fcs_concatenated)

    fc1 = Dense(2000, activation='relu', name='fc6')(drop1)
    drop2 = Dropout(0.5)(fc1)

    fc2 = Dense(2000, activation='relu', name='fc7')(drop2)

    au_detection_fc8 = Dense(10, activation='sigmoid',
                             kernel_initializer='normal',
                             name=AU_OCCURRENCE_OUTPUT_NAME)(fc2)

    model = Model(inputs=[vgg_model.input, landmarks],
                  outputs=[au_detection_fc8])

    return model


def _region_operation(region, index):
    region = Lambda(lambda x: K.tf.image.resize_images(x, (6, 6)))(region)
    region = Convolution2D(16, (3, 3), padding='valid',
                           kernel_initializer='normal',
                           activation='relu')(region)

    region = Flatten()(region)
    region = Dense(100, kernel_initializer='normal',
                   activation='relu')(region)

    return region

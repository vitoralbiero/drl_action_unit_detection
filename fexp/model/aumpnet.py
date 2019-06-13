from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers.merge import Concatenate, Add
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from fexp.layers import RegionLayer
from . import losses


INPUT_NAME = 'images'
POSE_OUTPUT_NAME = 'pose'
INPUT_INTENSITY_WEIGHTS = 'intensity_weights'
AU_OCCURRENCE_OUTPUT_NAME = 'au_occurrence'
AU_INTENSITY_OUTPUT_NAME = 'au_intensity'

N_AUS_INTENSITY = 7
N_AUS_OCCURRENCE = 10


def compile(model, writer):
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    decay = 0.0005
    epsilon = 1e-08

    writer.log_hyper_param('learning_rate_initial', lr)
    writer.log_hyper_param('momentum', beta_1)
    writer.log_hyper_param('momentum2', beta_2)
    writer.log_hyper_param('decay', decay)
    writer.log_hyper_param('epsilon', epsilon)

    optmizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
                    decay=decay, epsilon=epsilon)

    loss = ['binary_crossentropy',
            losses.weighted_mean_squared_error(ignore_label=9.0),
            'categorical_crossentropy']

    loss_weights = [1.0, 1.0, 0.5]

    model.compile(optimizer=optmizer,
                  loss=loss,
                  loss_weights=loss_weights)


def model(shape):
    # network input
    data_input = Input(shape=shape, name='images')

    # initial convolution
    au_conv1 = Convolution2D(16, (5, 5), padding='valid',
                             kernel_initializer='he_normal')(data_input)
    au_prelu1 = PReLU()(au_conv1)

    # region split
    region_layer = RegionLayer()
    region_layer.split(au_prelu1, n_cols=4, n_rows=4)

    # region operations
    region_layer.add(_region_operation)

    # region concat
    region_concatenated = region_layer.concatenate_convolution()

    # add face to regions
    au_add = Add()([au_prelu1, region_concatenated])

    # main face convolutions
    au_norm_res = BatchNormalization()(au_add)
    au_relu_res = PReLU()(au_norm_res)
    au_pool1 = MaxPooling2D((2, 2), strides=(2, 2))(au_relu_res)

    au_conv2 = Convolution2D(32, (5, 5), padding='valid',
                             kernel_initializer='he_normal')(au_pool1)
    au_relu2 = PReLU()(au_conv2)
    au_pool2 = MaxPooling2D((2, 2), strides=(2, 2))(au_relu2)
    au_conv3 = Convolution2D(64, (3, 3), padding='valid',
                             kernel_initializer='he_normal')(au_pool2)
    au_relu3 = PReLU()(au_conv3)

    # separated face branch convolutions
    au_face_pool1 = MaxPooling2D((2, 2), strides=(2, 2))(au_prelu1)
    au_face_conv1 = Convolution2D(32, (3, 3),
                                  kernel_initializer='he_normal',
                                  padding='valid')(au_face_pool1)
    au_face_relu1 = PReLU()(au_face_conv1)
    au_face_conv2 = Convolution2D(48, (3, 3),
                                  kernel_initializer='he_normal',
                                  padding='valid')(au_face_relu1)
    au_face_relu2 = PReLU()(au_face_conv2)
    au_face_pool2 = MaxPooling2D((2, 2), strides=(2, 2))(au_face_relu2)
    au_face_conv3 = Convolution2D(64, (3, 3),
                                  kernel_initializer='he_normal',
                                  padding='valid')(au_face_pool2)
    au_face_relu3 = PReLU()(au_face_conv3)

    # concatenate branches
    au_pose_concat = Concatenate(axis=1)([au_relu3, au_face_relu3])

    # flatten the convolutions
    flatten = Flatten()(au_pose_concat)

    # au fully connected layers
    au_fc6 = Dense(2000, kernel_initializer='he_normal')(flatten)
    au_relu6 = PReLU()(au_fc6)
    au_drop6 = Dropout(0.2)(au_relu6)
    au_fc7 = Dense(2000, kernel_initializer='normal')(au_drop6)
    au_relu7 = PReLU()(au_fc7)

    # detection output
    au_detection_fc8 = Dense(10, activation='sigmoid',
                             kernel_initializer='normal',
                             name=AU_OCCURRENCE_OUTPUT_NAME)(au_relu7)

    # intensity output
    au_intensity_fc8 = Dense(7, activation='sigmoid',
                             kernel_initializer='normal',
                             name='au_intensity_fc8')(au_relu7)

    au_intensity_scaled = Lambda(
        lambda x: x * 5.0, name=AU_INTENSITY_OUTPUT_NAME)(au_intensity_fc8)

    # pose fully connected layers
    pose_fc6 = Dense(160, kernel_initializer='normal')(flatten)
    pose_relu6 = PReLU()(pose_fc6)
    pose_drop6 = Dropout(0.5)(pose_relu6)
    pose_fc7 = Dense(160, kernel_initializer='normal')(pose_drop6)
    pose_relu7 = PReLU()(pose_fc7)
    pose_fc8_n = Dense(9, activation='softmax',
                       kernel_initializer='normal',
                       name=POSE_OUTPUT_NAME)(pose_relu7)

    model = Model(inputs=[data_input],
                  outputs=[au_detection_fc8, au_intensity_scaled,
                  pose_fc8_n])

    return model


def _region_operation(region):
    region = BatchNormalization()(region)
    region = PReLU()(region)
    region = Convolution2D(16, (3, 3), padding='same',
                           kernel_initializer='he_normal')(region)

    return region

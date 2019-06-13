from keras.engine import Model
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace
from keras.optimizers import SGD


INPUT_NAME = 'images'
AU_OCCURRENCE_OUTPUT_NAME = 'au_occurrence'
N_AUS_OCCURRENCE = 10


def compile(model, writer):
    lr = 0.001
    beta_1 = 0.9
    beta_2 = 0.999
    decay = 1e-6
    epsilon = 1e-08

    writer.log_hyper_param('learning_rate_initial', lr)
    writer.log_hyper_param('momentum', beta_1)
    writer.log_hyper_param('momentum2', beta_2)
    writer.log_hyper_param('decay', decay)
    writer.log_hyper_param('epsilon', epsilon)

    # optmizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2,
    #                 decay=decay, epsilon=epsilon)

    optmizer = SGD(lr=lr, decay=decay, momentum=beta_1)

    loss = ['binary_crossentropy']

    loss_weights = [1.0]

    model.compile(optimizer=optmizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=['acc'])


def model(shape, weights='vggface'):
    vgg_model = VGGFace(include_top=False,
                        input_shape=(224, 224, 3),
                        weights=weights)
    last_layer = vgg_model.get_layer('pool5').output

    x = Flatten(name='flatten')(last_layer)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)

    au_detection_fc8 = Dense(10, activation='sigmoid',
                             kernel_initializer='normal',
                             name=AU_OCCURRENCE_OUTPUT_NAME)(x)

    model = Model(vgg_model.input, outputs=[au_detection_fc8])

    return model

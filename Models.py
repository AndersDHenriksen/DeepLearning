from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten, Input
from tensorflow.keras.applications import ResNet50V2, MobileNetV2


def standard_cnn(config, classes):
    # 1.  Conv => Relu = > Pool
    input = Input(shape=config.input_shape)
    X = Conv2D(48, (7, 7), activation='relu')(input)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # 2. Conv => Relu = > Pool
    X = Conv2D(96, (5, 5), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # 3. Conv => Relu = > Pool
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # 4. Conv => Relu = > Pool
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # 5. Conv => Relu = > Pool
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # 6. Conv => Relu = > Pool
    X = Conv2D(256, (3, 3), activation='relu')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    # 7. Flat => Dropout
    X = Flatten()(X)
    X = Dropout(0.3)(X)

    # 8. FC => Relu => Dropout
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)

    # 9. FC => Relu
    X = Dense(16, activation='relu')(X)

    # 10. FC => Softmax
    output = Dense(classes, activation='softmax')(X)

    return Model(input, output)


def tl_resnet50(config, classes):
    # Build transfer learning network
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=config.input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Build new top
    X = base_model.output
    X = MaxPooling2D((7, 7))(X)  # Alternative X = DepthwiseConv2D((7, 7), activation='relu')(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax' if classes > 1 else 'sigmoid')(X)
    model = Model(inputs=base_model.input, outputs=X)
    model.is_in_warmup = True
    return model


def tl_mobilenet2(config, classes):
    # Build transfer learning network
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=config.input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    # Build new top
    X = base_model.output
    X = MaxPooling2D((7, 7))(X)  # Alternative X = DepthwiseConv2D((7, 7), activation='relu')(X)
    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax' if classes > 1 else 'sigmoid')(X)
    model = Model(inputs=base_model.input, outputs=X)
    model.is_in_warmup = True
    return model

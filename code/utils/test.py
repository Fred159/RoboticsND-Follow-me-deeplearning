def fcn_model(inputs, num_classes):
    # Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    print("Inputs  shape:", inputs.shape, "  \tImage Size in Pixels")

    encoder01 = encoder_block(inputs, filters=32, strides=2)
    print("encoder01 shape:", encoder01.shape, "  \tEncoder Block 1")

    encoder02 = encoder_block(encoder01, filters=64, strides=2)
    print("encoder02 shape:", encoder02.shape, "  \tEncoder Block 2")

    encoder03 = encoder_block(encoder02, filters=128, strides=2)
    print("encoder03 shape:", encoder03.shape, "\tEncoder Block 3")

    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_1x1 = conv2d_batchnorm(encoder03, filters=256, kernel_size=1, strides=1)
    print("conv_1x1 shape:", conv_1x1.shape, "\t1x1 Conv Layer")

    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder01 = decoder_block(conv_1x1, encoder02, filters=128)
    print("decoder01 shape:", decoder01.shape, "\tDecoder Block 1")

    decoder02 = decoder_block(decoder01, encoder01, filters=64)
    print("decoder02 shape:", decoder02.shape, "  \tDecoder Block 2")

    decoder03 = decoder_block(decoder02, inputs, filters=32)
    print("decoder03 shape:", decoder03.shape, "\tDecoder Block 3")

    # The function returns the output layer of your model. "decoder03" is the final layer obtained from the last decoder_block()

    # print("Outputs shape:", outputs.shape, "\tOutput Size in Pixel")

    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder03)


def fcn_model(inputs, num_classes):
    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    # conv2d 1st layer , input size is [,160,160,?]
    encoder1 = encoder_block(inputs, 32, 2)
    # conv2d 2nd layer, input size is [,80,80,32]
    encoder2 = encoder_block(encoder1, 64, 2)
    # conv2d 3rd layer, input size is [,40,40,64]
    encoder3 = encoder_block(encoder2, 128, 2)
    encoder4 = encoder_block(encoder3, 256, 2)
    # now, the input size is [,20,20,128], next step is 1x1 convolution layer
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_1x1 = conv2d_batchnorm(encoder4, 256, 1, 1)
    # after conv_1x1 process, the tensor size become [,20,20,128]
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder0 = decoder_block(conv_1x1, encoder4, 256)
    decoder1 = decoder_block(decoder0, encoder2, 128)
    decoder2 = decoder_block(decoder1, encoder1, 64)
    decoder3 = decoder_block(decoder2, inputs, 32)

    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(decoder3)
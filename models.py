import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy

def define_model_experimental(reg, conv_number, column_range, offset, 
    initial_height, dice_value):
    """Neural Network model implementation using Keras + Tensorflow."""

    # Calculating the channel dimensions given the board dynamics
    height = column_range[1] - column_range[0] + 1
    longest_column = (column_range[1] // 2) + 1
    width = initial_height + offset * (longest_column - column_range[0])
    # Calculating total number of actions possible (does not take into
    # consideration duplicate actions. Ex.: (2,3) and (3,2))
    temp = len(list(range(2, dice_value * 2 + 1)))
    n_actions = (temp*(1+temp))//2 + temp + 2

    n_channels = 6
    
    state_channels = Input(
                            shape = (n_channels, height, width), 
                            name='States_Channels_Input'
                            )
    valid_actions_dist = Input(
                                shape = (n_actions,), 
                                name='Valid_Actions_Input'
                                )

    zeropadding = keras.layers.ZeroPadding2D((2, 2))(state_channels)
    conv = Conv2D(
                    96, 
                    (4, 4), 
                    padding = "valid", 
                    kernel_initializer = 'glorot_normal', 
                    kernel_regularizer = regularizers.l2(reg), 
                    activation = 'relu', 
                    name = 'Conv_Layer'
                    )(zeropadding)
    conv2 = Conv2D(
                    96, 
                    (2, 2), 
                    padding = "valid", 
                    kernel_initializer = 'glorot_normal',
                    kernel_regularizer = regularizers.l2(reg), 
                    activation = 'relu', 
                    name = 'Conv_Layer2'
                    )(conv)
    batch1 = keras.layers.BatchNormalization()(conv2)
    act1 = keras.layers.Activation("relu")(batch1)
    flat = Flatten(name='Flatten_Layer')(act1)

    # Merge of the flattened channels (after pooling) and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_1'
                                )(merge)
    hidden_fc_prob_dist_2 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_2'
                                )(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(
                        n_actions, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'softmax', 
                        name = 'Output_Dist'
                        )(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(
                        100, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'relu', 
                        name = 'FC_Value_1'
                        )(flat)
    output_value = Dense(
                        1, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'tanh', 
                        name = 'Output_Value'
                        )(hidden_fc_value_1)

    model = Model(
                    inputs=[state_channels, valid_actions_dist], 
                    outputs=[output_prob_dist, output_value]
                    )

    model.compile(
                loss=['categorical_crossentropy','mean_squared_error'],
                optimizer='adam', 
                metrics={'Output_Dist':'categorical_crossentropy', 
                            'Output_Value':'mean_squared_error'},
                loss_weights = [1, 1])
    return model 

def define_model(reg, conv_number, column_range, offset, 
    initial_height, dice_value):
    """Neural Network model implementation using Keras + Tensorflow."""

    # Calculating the channel dimensions given the board dynamics
    height = column_range[1] - column_range[0] + 1
    longest_column = (column_range[1] // 2) + 1
    width = initial_height + offset * (longest_column - column_range[0])
    # Calculating total number of actions possible (does not take into
    # consideration duplicate actions. Ex.: (2,3) and (3,2))
    temp = len(list(range(2, dice_value * 2 + 1)))
    n_actions = (temp*(1+temp))//2 + temp + 2

    n_channels = 6

    state_channels = Input(
                        shape = (n_channels, height, width), 
                        name = 'States_Channels_Input'
                        )
    valid_actions_dist = Input(
                            shape = (n_actions,), 
                            name='Valid_Actions_Input'
                            )

    conv = Conv2D(
                filters = 10, 
                kernel_size = 2, 
                kernel_initializer = 'glorot_normal', 
                kernel_regularizer = regularizers.l2(reg), 
                activation = 'relu', 
                name = 'Conv_Layer'
                )(state_channels)
    if conv_number == 2:
        conv2 = Conv2D(
                    filters = 10, 
                    kernel_size = 2, 
                    kernel_initializer = 'glorot_normal',
                    kernel_regularizer = regularizers.l2(reg), 
                    activation = 'relu', 
                    name = 'Conv_Layer2'
                    )(conv)
    if conv_number == 1:
        flat = Flatten(name='Flatten_Layer')(conv)
    else:
        flat = Flatten(name='Flatten_Layer')(conv2)

    # Merge of the flattened channels and the valid action
    # distribution. Used only as input in the probability distribution head.
    merge = concatenate([flat, valid_actions_dist])

    #Probability distribution over actions
    hidden_fc_prob_dist_1 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_1'
                                )(merge)
    hidden_fc_prob_dist_2 = Dense(
                                100, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'relu', 
                                name = 'FC_Prob_2'
                                )(hidden_fc_prob_dist_1)
    output_prob_dist = Dense(
                                n_actions, 
                                kernel_initializer = 'glorot_normal', 
                                kernel_regularizer = regularizers.l2(reg), 
                                activation = 'softmax', 
                                name = 'Output_Dist'
                                )(hidden_fc_prob_dist_2)
    
    #Value of a state
    hidden_fc_value_1 = Dense(
                            100, 
                            kernel_initializer = 'glorot_normal', 
                            kernel_regularizer = regularizers.l2(reg), 
                            activation = 'relu', 
                            name = 'FC_Value_1'
                            )(flat)
    hidden_fc_value_2 = Dense(
                            100, 
                            kernel_initializer = 'glorot_normal', 
                            kernel_regularizer = regularizers.l2(reg), 
                            activation = 'relu', 
                            name = 'FC_Value_2'
                            )(hidden_fc_value_1)
    output_value = Dense(
                        1, 
                        kernel_initializer = 'glorot_normal', 
                        kernel_regularizer = regularizers.l2(reg), 
                        activation = 'tanh', 
                        name = 'Output_Value'
                        )(hidden_fc_value_2)

    model = Model(
                inputs=[state_channels, valid_actions_dist], 
                outputs=[output_prob_dist, output_value]
                )

    model.compile(
                loss=['categorical_crossentropy','mean_squared_error'],
                optimizer='adam', 
                metrics={'Output_Dist':'categorical_crossentropy', 
                            'Output_Value':'mean_squared_error'},
                loss_weights = [1, 1])

    return model  
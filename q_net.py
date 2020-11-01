def build_dqn_model(lr):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.compat.v1.keras.backend import set_session
    set_session(tf.compat.v1.Session())
    model = keras.Sequential([
        keras.layers.Dense(128, input_shape = (2, ), activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3, activation='linear')])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    #model.save('q_net_model')
    return model



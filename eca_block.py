C = 256
r = 16
gamma, beta = 2, 1
k_size = math.log(C, 2) / gamma + beta / gamma
k_size = round(k_size) if round(k_size) % 2 == 1 else \
         round(k_size) - 1 if abs(k_size - round(k_size) + 1) < abs(k_size - round(k_size) - 1) else \
         round(k_size) + 1
         # set k_size to nearest odd
# Conv layers
x = KL.Conv2D(256, (3, 3), padding="same")(x)
x1 = KL.Activation('relu')(x)

x = KL.GlobalAveragePooling2D()(x1)
x = KL.Reshape((C, 1))(x)
x = KL.Conv1D(1, kernel_size=k_size, padding="same", activation='sigmoid')(x)
x = KL.Reshape((1, 1, C))(x)
x = KL.Multiply()([x1, x])
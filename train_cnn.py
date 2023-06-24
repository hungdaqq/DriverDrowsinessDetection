import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ProgbarLogger

# Set the input image size and number of classes
input_shape = (128, 128, 3)
num_classes = 2
num_epochs = 10

# Create the EfficientNet model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Prepare your data using an ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/home/hung/DL2023/DriverDrowsinessDetection/classify/',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    '/home/hung/DL2023/DriverDrowsinessDetection/val/',
    target_size=(input_shape[0], input_shape[1]),
    batch_size=32,
    class_mode='categorical')

progbar_logger = ProgbarLogger()

# Train the model
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=num_epochs,
          validation_data=val_generator,
          validation_steps=len(val_generator),
          callbacks=[progbar_logger])

# Save the model as a .pb file
pb_path = './model.pb'
tf.saved_model.save(model, pb_path)

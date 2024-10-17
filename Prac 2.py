import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 1: Data Preparation with Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    r"C:\Users\NiravG\Downloads\archive (7)\chest_xray\train",  # Path to training data
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    r"C:\Users\NiravG\Downloads\archive (7)\chest_xray\val",  # Path to validation data
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    r"C:\Users\NiravG\Downloads\archive (7)\chest_xray\test",  # Path to test data
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Step 2: Model Architecture - ResNet50 with Fine-tuning

# Load the pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[:-10]:  # Unfreeze last 10 layers
    layer.trainable = False
for layer in base_model.layers[-10:]:  # Fine-tune last 10 layers
    layer.trainable = True

# Add custom layers on top of ResNet50
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)  # Add Batch Normalization
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)  # Add Batch Normalization
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
              loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Step 3: Learning Rate Scheduler
def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.1  # Reduce learning rate after 10 epochs
    return lr

# Step 4: Model Training with Learning Rate Scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=30,  # Increase number of epochs to allow more training
    callbacks=[lr_scheduler]  # Include learning rate scheduler
)

# Step 5: Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Predict on the test set
y_pred = model.predict(test_generator)
y_pred = (y_pred > 0.5).astype(int)  # Threshold for binary classification

# Get ground truth labels
y_true = test_generator.classes

# Step 6: Classification Report and Confusion Matrix
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

print('Confusion Matrix:')
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Step 7: Plot training and validation accuracy and loss
import matplotlib.pyplot as plt

# Retrieve the accuracy and loss from the history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(10, 5))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
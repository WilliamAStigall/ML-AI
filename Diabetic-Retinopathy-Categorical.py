import os
import shutil
import tensorflow as tf
import keras
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import KFold
import keras_tuner
import imghdr
import datetime
import matplotlib.pyplot as plt

# Set the GPU device
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_dataset(dataframe, batch_size=32, image_size=(224, 224), shuffle=True):
    indexes = np.arange(len(dataframe))
    if shuffle:
        np.random.shuffle(indexes)
    
    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except tf.errors.InvalidArgumentError:
            return None # Return None if the image is not a valid JPEG file
        image = tf.image.resize(image, image_size)
        return image

    def is_valid_image(image):
        return image is not None # Check if the image is not None

    paths = dataframe["path"].values[indexes]
    levels = dataframe["level"].values[indexes]

    dataset = tf.data.Dataset.from_tensor_slices((paths, levels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
    dataset = dataset.filter(lambda x, y: tf.py_function(is_valid_image, [x], tf.bool)) # Skip any images that raise errors
    dataset = dataset.filter(lambda x, y: tf.logical_or(tf.equal(y, '0'), tf.not_equal(y, '0')))
    dataset = dataset.map(lambda x,y: (x ,tf.one_hot(tf.strings.to_number(y, out_type=tf.int32), depth=5)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset




    

class ResetWeights(tf.keras.callbacks.Callback):
    def __init__(self,model):
        self.model = model
        #Save initial weights
        self.initial_weights = model.get_weights()
        
    def on_train_begin(self, logs=None):
        self.model.set_weights(self.initial_weights)
        
# Define a custom F1 score function
def f1_score(y_true, y_pred):
  # Calculate precision and recall
  precision = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32)), tf.reduce_sum(tf.cast(y_pred, tf.float32)))
  recall = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32)), tf.reduce_sum(tf.cast(y_true, tf.float32)))
  
  # Calculate F1 score
  f1_score = 2 * (precision * recall) / (precision + recall)
  
  # Return a tensor value
  return f1_score    
    




# Define the HyperModel subclass
class MyHyperModel(keras_tuner.HyperModel):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def build(self, hp):
        # Define the KFold cross-validation with 5 splits
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize a list to store the validation accuracies for each fold
        val_accs = []
        
        
        augmentation = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,  # Random rotation between -10 and 10 degrees
            width_shift_range=0.1,  # Random horizontal shift by 10% of the image width
            height_shift_range=0.1,  # Random vertical shift by 10% of the image height
            shear_range=0.1,  # Shear transformations with a maximum shear of 0.1
            zoom_range=0.1,  # Random zoom between 0.9 and 1.1
            horizontal_flip=True,  # Randomly flip images horizontally
            vertical_flip=False,  # Do not flip images vertically
            
        )
        # Loop over each fold
        for train_index, val_index in kf.split(self.dataframe):
            # Split the dataframe into train and validation sets
            train_df = self.dataframe.iloc[train_index]
            val_df = self.dataframe.iloc[val_index]

            # Create data generators for each set
            train_gen = create_dataset(train_df)
            val_gen = create_dataset(val_df)
            # Define the hyperparameters to tune
            lr = hp.Float("learning_rate", min_value=0.0001, max_value=0.1, sampling="log")
            dropout = hp.Float("dropout_rate", min_value = 0.1, max_value = 0.5, sampling = "log")
            dense_1 = hp.Int("dense_units_1", min_value=64, max_value=512, step=32)
            dense_2 = hp.Int("dense_units_2", min_value=64, max_value=512, step=32)
            dense_3 = hp.Int("dense_units_3", min_value=64, max_value=512, step=32)

            # Build the model using InceptionV3 as base model
            base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Dense(dense_1, activation="relu")(x)
            x = tf.keras.layers.Dense(dense_2, activation="relu")(x)
            x = tf.keras.layers.Dense(dense_3, activation='relu')(x)
            output = tf.keras.layers.Dense(5, activation="softmax")(x)
            model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

            # UnFreeze the base model layers
            for layer in base_model.layers:
                layer.trainable = False
            
            
            # Compile the model with multiple metrics for categorical classification
            model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), 
                          loss="categorical_crossentropy", 
                          metrics=["accuracy",
                                   tf.keras.metrics.Precision(),
                                   tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC(),
                                   f1_score
                            
                          ])                 
            # Print the model summary
            model.summary()
            reset_weights = ResetWeights(model)
            # Train the model with early stopping callback
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath="hypermodel.h5", # The path to save the model
                    monitor="val_accuracy", # The metric to monitor
                    mode="max", # The mode to compare the metric values
                    save_best_only=True, # Only save the model with the best metric value
                    save_freq="epoch" # Save the model at the end of each epoch
                )
            # Create a log directory with a timestamp
            log_dir = "logs/hypermodel/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # Create a TensorBoard callback
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            history = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
                                                                                          reset_weights, tensorboard_callback,checkpoint])
                        # Evaluate the model on the validation set

            # Calculate specificity (True Negative Rate) for the validation set
            predictions = model.predict(val_gen)
            

            # Get the best validation accuracy for this fold
            val_loss, val_accuracy, val_precision, val_recall, val_auc, val_f1 = model.evaluate(val_gen)
           
            
            # Append the best validation accuracy to the list
            val_accs.append(val_accuracy)
        model.save("DR_Categorical_HyperModel.h5")
        # Return the mean validation accuracy across all folds as the objective value
        return np.mean(val_accs)
    

    
    
def transfer_learning(dataframe, hypermodel):
    
    best_model = hypermodel
    base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False, input_shape=(224,224,3), pooling='avg')
    model = tf.keras.models.Model(inputs=base_model.input, outputs=best_model(base_model.output))
                                  
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss="categorical_crossentropy",
                  metrics=['accuracy',
                  tf.keras.metrics.Precision(),
                  tf.keras.metrics.Recall(),
                  tf.keras.metrics.AUC(),
                  f1_score
                  ])
    data_gen = create_dataset(dataframe)
    # Create a log directory with a timestamp
    log_dir = "logs/transfer_learning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create a TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
                    filepath="transfer_learning.h5", # The path to save the model
                    monitor="val_accuracy", # The metric to monitor
                    mode="max", # The mode to compare the metric values
                    save_best_only=True, # Only save the model with the best metric value
                    save_freq="epoch" # Save the model at the end of each epoch
                )
    history = model.fit(data_gen, epochs=40, callbacks =  [tf.keras.callbacks.ReduceLROnPlateau,tensorboard_callback,checkpoint])
    model.save('DR_Binar_TL.h5')
    return model

def create_dataset_no_batch(dataframe,  image_size=(224, 224), shuffle=True):
    indexes = np.arange(len(dataframe))
    if shuffle:
        indexes = np.random.permutation(indexes)
        print(len(dataframe))
    
    def preprocess_image(image_path):
        image = tf.io.read_file(image_path)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except tf.errors.InvalidArgumentError:
            return None # Return None if the image is not a valid JPEG file
        image = tf.image.resize(image, image_size)
        return image

    def is_valid_image(image):
        return image is not None # Check if the image is not None

    paths = dataframe["path"].values[indexes]
    levels = dataframe["level"].values[indexes]

    dataset = tf.data.Dataset.from_tensor_slices((paths, levels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y))
     # Skip any images that raise errors
    dataset = dataset.filter(lambda x, y: tf.logical_or(tf.equal(y, '0'), tf.not_equal(y, '0')))
    dataset = dataset.map(lambda x,y: (x ,tf.one_hot(tf.strings.to_number(y, out_type=tf.int32), depth=5)))
    # Calculate the length of the dataset manually
    dataset_length = sum(1 for _ in dataset)

    # Print the length of the dataset
    print("Length of dataset:", dataset_length)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset





def predict_testset(dataframe, model, log_dir):
    predictions = []
    labels = []

    for image, label in create_dataset_no_batch(dataframe):
        image = tf.image.resize(image, (224, 224))  # Resize image to match model input shape
    
        # Check if the image has 3 dimensions, if not, expand the dimensions
        if len(image.shape) < 3:
            image = tf.expand_dims(image, axis=-1)

        image = tf.expand_dims(image, axis=0)  # Add batch dimension to the image
        prediction = model.predict(image)[0]
        predictions.append(prediction)
        labels.append(label)
    test_data = create_dataset(dataframe)
    

    # Convert predictions and labels to numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    predicted_labels = np.argmax(predictions, axis=1)
    

    

    predicted_labels = np.argmax(predictions, axis=1)
    results = pd.concat([dataframe['PId'], pd.DataFrame(predictions)], axis=1)
    # Create a TensorBoard callback and pass it to the model.evaluate()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.evaluate(test_data, callbacks=[tensorboard_callback])

    from sklearn.metrics import confusion_matrix
    converted_labels = np.argmax(labels, axis=1)
    # Load the true labels and predicted labels from the dataframe
    y_true = converted_labels
    y_pred = predicted_labels

    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar()

    # Add the labels and counts to the plot
    classes = ['0', '1', '2', '3', '4']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    # Show the plot
    plt.savefig('DR_Categorical_ConfusionMatrix.png')
    return results

    
    
    
        
    
    
train_data_dir = r"C:\Users\_\Downloads\diabetic-retinopathy-detection\train\train"
train_labels_dir = r"C:\_\rstig\Downloads\diabetic-retinopathy-detection\trainLabels\trainLabels.csv"
test_data_dir = r"C:\Users\_\Downloads\diabetic-retinopathy-detection\test\test"
test_labels_dir = r"C:\Users\_\Downloads\diabetic-retinopathy-detection\retinopathy_solution.csv"
# Set the display options

train_dataframe = pd.read_csv(train_labels_dir, sep=',')
test_dataframe = pd.read_csv(test_labels_dir, sep=',')

# Update the 'path' column with the complete file path
train_dataframe["PId"] = train_dataframe['image'].map(lambda x: x.split('_')[0])
train_dataframe["path"] = train_dataframe['image'].map(lambda x: os.path.join(train_data_dir,'{}.jpeg'.format(x)))

test_dataframe["PId"] = test_dataframe['image'].map(lambda x: x.split('_')[0])
test_dataframe["path"] = test_dataframe['image'].map(lambda x: os.path.join(test_data_dir,'{}.jpeg'.format(x)))

# Convert the 'level' column to string
train_dataframe['level'] = train_dataframe['level'].astype(str)
test_dataframe['level'] = test_dataframe['level'].astype(str)
        

num_images_found = train_dataframe[train_dataframe["path"].map(os.path.exists)].shape[0]
total_images = train_dataframe.shape[0]
print(num_images_found, "training images found of ", total_images, "total training images")

num_images_found = test_dataframe[test_dataframe["path"].map(os.path.exists)].shape[0]
total_images = test_dataframe.shape[0]
print(num_images_found, "testing images found of ", total_images, "total testing images")

# Replace any empty strings with NaN values
train_dataframe['level'] = train_dataframe['level'].replace('', np.nan)
test_dataframe['level'] = test_dataframe['level'].replace('', np.nan)

# Drop any rows that contain NaN values
train_dataframe.dropna(inplace=True)
test_dataframe.dropna(inplace=True)

# Check the dataframe after dropping empty strings and NaN values
print(train_dataframe)
print(test_dataframe)



train_gen = create_dataset(train_dataframe)
test_gen = create_dataset(test_dataframe)





print("num train dataframe levels", train_dataframe['level'].value_counts())
print("num test_dataframe levels", test_dataframe['level'].value_counts())



hypermodel = MyHyperModel(train_dataframe)

objective = keras_tuner.Objective("val_accuracy+val_precision",direction="max")
tuner = keras_tuner.BayesianOptimization(hypermodel, objective=objective, max_trials=5, seed=42)

#Perform the optimization

model = transfer_learning(train_dataframe,hypermodel)

hypermodel = keras.models.load_model( custom_objects={"f1_score": f1_score},filepath=r"C:\Users\_\VS_Code_Projects\DR_Categorical_HyperModel.h5")

log_dir = r"C:\Users\_\VS_Code_Projects\logs\Test_Predictions_Cat"

results = predict_testset(test_dataframe,hypermodel,log_dir)

results.to_csv('predictions_probabilities.csv', index = False)


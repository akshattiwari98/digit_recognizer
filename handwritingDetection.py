import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Character set for encoding/decoding
CHAR_LIST = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
RECORDS_COUNT = 25000
BATCH_SIZE = 5
EPOCHS = 20
INPUT_HEIGHT = 32
INPUT_WIDTH = 128

class HandwritingDetection:
    def __init__(self, char_list=CHAR_LIST, max_label_len=0):
        self.char_list = char_list
        self.max_label_len = max_label_len
        self.train_data = {'images': [], 'labels': [], 'input_length': [], 'label_length': [], 'original_text': []}
        self.valid_data = {'images': [], 'labels': [], 'input_length': [], 'label_length': [], 'original_text': []}
    
    def process_image(self, img):
        """Converts image to shape (32, 128, 1) & normalize"""
        try:
            w, h = img.shape
            new_w = 32
            new_h = int(h * (new_w / w))
            img = cv2.resize(img, (new_h, new_w))
            w, h = img.shape
            img = img.astype('float32')
            
            if w < 32:
                add_zeros = np.full((32 - w, h), 255)
                img = np.concatenate((img, add_zeros))
                w, h = img.shape
            
            if h < 128:
                add_zeros = np.full((w, 128 - h), 255)
                img = np.concatenate((img, add_zeros), axis=1)
                w, h = img.shape
            
            if h > 128 or w > 32:
                img = cv2.resize(img, (128, 32))
            
            img = cv2.subtract(255, img)
            img = np.expand_dims(img, axis=2)
            img = img / 255
            
            return img
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def encode_to_labels(self, txt):
        """Encode text to label indices"""
        return [self.char_list.index(char) for char in txt if char in self.char_list]
    
    def load_data(self, filepath, is_validation=False, test_split=0.1):
        """Load and process data from file"""
        try:
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            
            processed_img = self.process_image(img)
            if processed_img is None:
                return False
            
            return processed_img
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def prepare_labels(self):
        """Prepare padded labels for training"""
        train_labels = pad_sequences(
            self.train_data['labels'],
            maxlen=self.max_label_len,
            padding='post',
            value=len(self.char_list)
        )
        
        valid_labels = pad_sequences(
            self.valid_data['labels'],
            maxlen=self.max_label_len,
            padding='post',
            value=len(self.char_list)
        )
        
        return train_labels, valid_labels
    
    def convert_to_arrays(self):
        """Convert lists to numpy arrays"""
        for key in self.train_data:
            if key != 'original_text':
                self.train_data[key] = np.asarray(self.train_data[key])
                self.valid_data[key] = np.asarray(self.valid_data[key])
    
    def build_model(self):
        """Build CRNN model architecture"""
        inputs = Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 1))
        
        # Convolutional layers
        conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
        
        conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
        pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
        
        conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
        conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
        pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
        
        conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)
        batch_norm_5 = BatchNormalization()(conv_5)
        
        conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
        batch_norm_6 = BatchNormalization()(conv_6)
        pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
        
        conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)
        squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
        
        # LSTM layers
        blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(squeezed)
        blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))(blstm_1)
        
        outputs = Dense(len(self.char_list) + 1, activation='softmax')(blstm_2)
        
        return Model(inputs, outputs), outputs, inputs
    
    def train_model(self, train_images, train_labels, valid_images, valid_labels, 
                    train_input_length, train_label_length, valid_input_length, valid_label_length):
        """Train the model with CTC loss"""
        act_model, outputs, inputs = self.build_model()
        
        the_labels = Input(name='the_labels', shape=[self.max_label_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        
        def ctc_lambda_func(args):
            y_pred, labels, input_length, label_length = args
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, the_labels, input_length, label_length])
        model = Model(inputs=[inputs, the_labels, input_length, label_length], outputs=loss_out)
        
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='sgd', metrics=['accuracy'])
        
        checkpoint = ModelCheckpoint(filepath='best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True)
        
        history = model.fit(
            x=[train_images, train_labels, train_input_length, train_label_length],
            y=np.zeros(len(train_images)),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([valid_images, valid_labels, valid_input_length, valid_label_length], 
                           [np.zeros(len(valid_images))]),
            verbose=1,
            callbacks=[checkpoint]
        )
        
        return model, act_model, history
    
    @staticmethod
    def plot_metrics(history):
        """Plot training metrics"""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs, loss, 'b', label='Train Loss')
        ax1.plot(epochs, val_loss, 'r', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(epochs, acc, 'b', label='Train Accuracy')
        ax2.plot(epochs, val_acc, 'r', label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
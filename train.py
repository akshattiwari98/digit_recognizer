import numpy as np
import os
from datetime import datetime
from handwriting_detection_optimized import HandwritingDetection

def main():
    """Train and save the handwriting detection model"""
    
    # Initialize detector
    detector = HandwritingDetection()
    detector.max_label_len = 25  # Set based on your data
    
    # Load your prepared data here
    # Example: Load from pickle, numpy files, or generate from notebooks
    train_images = np.load('data/train_images.npy')
    train_labels = np.load('data/train_labels.npy')
    train_input_length = np.load('data/train_input_length.npy')
    train_label_length = np.load('data/train_label_length.npy')
    
    valid_images = np.load('data/valid_images.npy')
    valid_labels = np.load('data/valid_labels.npy')
    valid_input_length = np.load('data/valid_input_length.npy')
    valid_label_length = np.load('data/valid_label_length.npy')
    
    print(f"Training data shape: {train_images.shape}")
    print(f"Validation data shape: {valid_images.shape}")
    
    # Train model
    print("\n🚀 Starting training...")
    model, act_model, history = detector.train_model(
        train_images, train_labels,
        valid_images, valid_labels,
        train_input_length, train_label_length,
        valid_input_length, valid_label_length
    )
    
    # Plot results
    print("\n📊 Plotting metrics...")
    HandwritingDetection.plot_metrics(history)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save the model with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/handwriting_model_{timestamp}.hdf5'
    act_model.save(model_path)
    print(f"\n✅ Model saved: {model_path}")
    
    # Also save weights separately
    weights_path = f'models/handwriting_weights_{timestamp}.h5'
    act_model.save_weights(weights_path)
    print(f"✅ Weights saved: {weights_path}")
    
    return act_model, history

if __name__ == '__main__':
    trained_model, training_history = main()
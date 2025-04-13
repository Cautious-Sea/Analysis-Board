#!/usr/bin/env python3
"""
Train a Chess Piece Recognition Model

This script trains a neural network to recognize chess pieces from images.
It uses TensorFlow/Keras and supports data augmentation to improve model performance.

Usage:
    python train_chess_model.py --dataset_path PATH_TO_DATASET [--epochs 50] [--batch_size 32]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define piece classes
PIECE_CLASSES = [
    'empty',  # 0
    'P', 'N', 'B', 'R', 'Q', 'K',  # White pieces (1-6)
    'p', 'n', 'b', 'r', 'q', 'k'   # Black pieces (7-12)
]

def create_model(num_classes=13, input_shape=(64, 64, 3)):
    """Create a model for chess piece classification using transfer learning."""
    # Use MobileNetV2 as the base model (lightweight and efficient)
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def prepare_data_generators(dataset_path, batch_size=32, img_size=(64, 64)):
    """Prepare data generators for training and validation."""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Load validation data
    validation_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train_model(dataset_path, output_dir='models', epochs=50, batch_size=32, img_size=(64, 64)):
    """Train the chess piece classifier model."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data generators
    train_generator, validation_generator = prepare_data_generators(
        dataset_path, batch_size, img_size
    )
    
    # Print class indices
    print("Class indices:", train_generator.class_indices)
    
    # Create the model
    model, base_model = create_model(
        num_classes=len(train_generator.class_indices),
        input_shape=(*img_size, 3)
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'chess_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train the model
    print("\n--- Starting initial training phase ---")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs // 2,  # First half of epochs for initial training
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Fine-tune the model (unfreeze some layers)
    print("\n--- Starting fine-tuning phase ---")
    # Unfreeze the top layers of the base model
    base_model.trainable = True
    
    # Freeze all the layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model again
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs // 2,  # Second half of epochs for fine-tuning
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[
            ModelCheckpoint(
                os.path.join(output_dir, 'chess_model_fine_tuned.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            early_stopping,
            reduce_lr
        ]
    )
    
    # Combine histories
    combined_history = {
        'accuracy': history.history['accuracy'] + history_fine.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'] + history_fine.history['val_accuracy'],
        'loss': history.history['loss'] + history_fine.history['loss'],
        'val_loss': history.history['val_loss'] + history_fine.history['val_loss']
    }
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(combined_history['accuracy'])
    plt.plot(combined_history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(combined_history['loss'])
    plt.plot(combined_history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    
    # Save the final model
    model.save(os.path.join(output_dir, 'chess_model_final.h5'))
    print(f"Model saved to {os.path.join(output_dir, 'chess_model_final.h5')}")
    
    # Evaluate the model
    evaluate_model(model, validation_generator, output_dir)
    
    return model

def evaluate_model(model, validation_generator, output_dir):
    """Evaluate the model and generate performance metrics."""
    # Get predictions
    validation_generator.reset()
    y_pred = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size + 1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get true labels
    y_true = validation_generator.classes[:len(y_pred_classes)]
    
    # Get class names
    class_indices = validation_generator.class_indices
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    
    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    
    # Save classification report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save some example predictions
    validation_generator.reset()
    batch_x, batch_y = next(validation_generator)
    predictions = model.predict(batch_x)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(15, len(batch_x))):
        plt.subplot(3, 5, i+1)
        plt.imshow(batch_x[i])
        true_label = class_names[np.argmax(batch_y[i])]
        pred_label = class_names[np.argmax(predictions[i])]
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'example_predictions.png'))

def create_dataset_structure(output_dir):
    """Create the dataset directory structure."""
    # Create main directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each piece type
    for piece_class in PIECE_CLASSES:
        os.makedirs(os.path.join(output_dir, piece_class), exist_ok=True)
    
    print(f"Created dataset directory structure at {output_dir}")
    print("Please add your images to the appropriate subdirectories:")
    for piece_class in PIECE_CLASSES:
        print(f"  - {os.path.join(output_dir, piece_class)}")

def main():
    """Main function to parse arguments and run the training."""
    parser = argparse.ArgumentParser(description="Train Chess Piece Recognition Model")
    parser.add_argument("--dataset_path", help="Path to the dataset directory")
    parser.add_argument("--create_dataset", help="Create dataset directory structure at the specified path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output_dir", default="models", help="Directory to save models and results")
    
    args = parser.parse_args()
    
    if args.create_dataset:
        create_dataset_structure(args.create_dataset)
        return 0
    
    if not args.dataset_path:
        print("Error: Please provide --dataset_path or --create_dataset")
        parser.print_help()
        return 1
    
    if not os.path.isdir(args.dataset_path):
        print(f"Error: {args.dataset_path} is not a directory")
        return 1
    
    # Check if the dataset has the expected structure
    expected_dirs = PIECE_CLASSES
    missing_dirs = [d for d in expected_dirs if not os.path.isdir(os.path.join(args.dataset_path, d))]
    
    if missing_dirs:
        print(f"Warning: The following expected directories are missing from the dataset:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("\nYou can create the dataset structure using --create_dataset")
        
        proceed = input("Do you want to proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return 1
    
    # Train the model
    train_model(
        args.dataset_path,
        args.output_dir,
        args.epochs,
        args.batch_size
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

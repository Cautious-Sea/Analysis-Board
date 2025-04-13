# Chess Position Detection with Custom Model Training

This project provides tools for detecting chess positions from images and outputting FEN notation. It includes scripts for:

1. Detecting chess positions from images
2. Collecting training data for chess piece recognition
3. Training a custom neural network model for improved accuracy

## Requirements

Install the required packages:

```bash
pip install numpy opencv-python tensorflow matplotlib pillow
```

## Quick Start

To detect a chess position from an image:

```bash
python chess_fen_detector.py path/to/chess_image.jpg
```

To use a trained model for better accuracy:

```bash
python chess_fen_detector.py path/to/chess_image.jpg --model path/to/model.h5
```

To visualize the detection process:

```bash
python chess_fen_detector.py path/to/chess_image.jpg --model path/to/model.h5 --visualize
```

## Training Your Own Model

### Step 1: Create Dataset Structure

First, create the dataset directory structure:

```bash
python train_chess_model.py --create_dataset chess_dataset
```

This will create a directory structure with subdirectories for each piece type:
- `empty` - Empty squares
- `P`, `N`, `B`, `R`, `Q`, `K` - White pieces
- `p`, `n`, `b`, `r`, `q`, `k` - Black pieces

### Step 2: Collect Training Data

You can collect training data from chess board images using the provided tool:

```bash
python collect_training_data.py --image_path path/to/chess_image.jpg --output_dir chess_dataset
```

This will:
1. Detect the chess board in the image
2. Extract the 64 squares
3. Display each square and let you label it
4. Save the labeled squares to the appropriate subdirectories

Repeat this process with multiple chess board images to build a comprehensive dataset.

### Step 3: Train the Model

Once you have collected enough training data, train the model:

```bash
python train_chess_model.py --dataset_path chess_dataset --epochs 50 --batch_size 32
```

This will:
1. Load the training data from the dataset directory
2. Train a neural network model using transfer learning
3. Save the trained model to the `models` directory
4. Generate performance metrics and visualizations

The training process includes:
- Initial training phase
- Fine-tuning phase
- Performance evaluation

### Step 4: Use the Trained Model

After training, you can use your custom model with the FEN detector:

```bash
python chess_fen_detector.py path/to/chess_image.jpg --model models/chess_model_fine_tuned.h5
```

## Tips for Better Results

### Data Collection
- Collect images of chess boards in different lighting conditions
- Include images with different chess piece styles
- Make sure to have enough examples of each piece type
- Include empty squares in your dataset

### Training
- Increase the number of epochs for better accuracy
- Adjust the batch size based on your available memory
- Use data augmentation to increase the effective size of your dataset
- Monitor the validation accuracy to prevent overfitting

### Detection
- Use good lighting when taking photos of chess boards
- Make sure the board is clearly visible in the image
- Try different angles if the detection is not accurate
- Use the visualization option to see how the detection is working

## How It Works

The detection process involves several steps:

1. **Board Detection**: The script first detects the chess board in the image using contour detection and perspective transformation.

2. **Square Extraction**: Once the board is detected, it extracts the 64 individual squares.

3. **Piece Classification**: Each square is classified using either:
   - The trained neural network model (if provided)
   - A simple color and shape-based approach (fallback)

4. **FEN Generation**: The detected pieces are converted to FEN notation, which is a standard way to represent chess positions.

## Using the FEN Notation

The FEN notation can be used to set up the position in your chess variant application:

1. Open the chess variant application
2. Click on 'Setup Position'
3. Click on 'Clear Board'
4. Set up the pieces according to the FEN notation
5. Click on 'Done Setup'

You can also view the position on lichess.org using the provided link.

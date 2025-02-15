
## Skin Lesion Classification Using SVM and VGG16 for HAM10000 Dataset
- This project implements a machine learning approach for skin lesion classification using KNN,  Support Vector Machines (SVM) and transfer learning with a pre-trained VGG16 model on the HAM10000 dataset.

**Project Overview**
- The project aims to classify skin lesions into various categories based on their visual features. Here's a breakdown of the approach:
- Data Preprocessing: Load and pre-process images from the HAM10000 dataset. This might involve resizing, normalization, and data augmentation.
- Feature Extraction: Utilize the pre-trained VGG16 model to extract high-level features from the preprocessed images. We freeze the weights of the VGG16 model and only train the final layers for classification.
- Classification:
  - Train a Support Vector Machine classifier and a KNN classifier on the extracted features to classify skin lesions into different categories.
- Evaluation: Evaluate the performance of the model using metrics like accuracy, precision, recall, and F1-score.

**Dependencies**
This project requires the following Python libraries:
- TensorFlow
- scikit-learn
- OpenCV (for image processing)
- NumPy
Note: Specific version requirements might be included in a requirements.txt file for easy installation.


Pre-trained weights for the VGG16 model might be downloaded automatically or provided within the project directory.
The project might include scripts for data visualization and analysis.

Contributing
We welcome contributions to this project! Feel free to create pull requests with improvements or bug fixes.

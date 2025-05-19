
# 🐾 PRODIGY ML TASK 3: Cat vs. Dog Image Classification with SVM 🐶



**🎯 Task:** Implement a support vector machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

**💾 Dataset:** [Kaggle Cats vs. Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)

**🖼️ Project Overview**

This repository demonstrates image classification using a Support Vector Machine (SVM) to distinguish between images of cats and dogs. The project utilizes popular Python libraries for image processing and machine learning:

* **OpenCV:** For loading and preprocessing images. 🖼️
* **NumPy:** For numerical operations and array manipulation. 🔢
* **Scikit-learn:** For implementing the SVM model and evaluation metrics. 🤖
* **Matplotlib & Seaborn:** For visualization, including plotting the confusion matrix. 📊

The steps involved in this project include:

1.  **Loading Images:** Reading images of cats and dogs.
2.  **Preprocessing:** Resizing images to a fixed size, converting them to grayscale, and flattening them into 1D arrays to be used as input for the SVM.
3.  **Training the SVM Model:** Using the preprocessed image data to train an SVM classifier.
4.  **Evaluation:** Assessing the performance of the trained SVM model on a test set.
5.  **Visualization:** Enhancing the understanding of the model's performance using a confusion matrix.

**📂 Project Structure**

PRODIGY_ML_TASK_3/
├── cats/                # Directory containing images of cats
├── dogs/                # Directory containing images of dogs
└── Task_3.ipynb         # Jupyter Notebook containing the Python script for loading, preprocessing, training, and evaluating the SVM model
└── README.md            # This README file


**⚙️ Requirements**

Make sure you have the following Python libraries installed:

```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn
🚀 Getting Started

Clone the repository: 💻

Bash

git clone [https://github.com/AJ22122003/PRODIGY_ML_TASK_3.git](https://github.com/AJ22122003/PRODIGY_ML_TASK_3.git)
cd PRODIGY_ML_TASK_3
Download the Dataset: 💾 Download the "train" dataset from the Kaggle Cats vs. Dogs competition and place the cats and dogs image folders inside the repository.

Explore the Jupyter Notebook: 📒 Open the Task_3.ipynb file using Jupyter Notebook or JupyterLab to follow the implementation details.

Bash

jupyter notebook Task_3.ipynb
# or
jupyter lab Task_3.ipynb
Run the Notebook Cells: Execute the cells in the notebook sequentially to:

Load and preprocess the cat and dog images.
Split the data into training and testing sets.
Train the SVM classifier.
Make predictions on the test set.
Evaluate the model's accuracy.
Visualize the confusion matrix to understand the classification results. 📊
🧠 Model

This project utilizes a Support Vector Machine (SVM), a powerful supervised learning algorithm used for classification. SVM works by finding the hyperplane that best separates the different classes in the feature space.

📈 Results

The SVM model achieves an accuracy of X% (as mentioned in the README content) on the test set, effectively distinguishing between images of cats and dogs. The confusion matrix provides a detailed breakdown of the model's performance, showing the counts of true positives, true negatives, false positives, and false negatives.

🤝 Contributions

[Optional: If you'd like to encourage contributions, add a section here explaining how others can contribute to the project.]

👨‍💻 Author

This project was created by [Ajinkya Kutarmare].

Explore the fascinating world of image classification with SVM! 🐾🐶

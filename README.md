# Malaria_detection
A convolution nerural network Model built from scratch to detect Malaria and deployed on web application build on streamlit.

The Demo video of the model running on Streamlit web application: https://drive.google.com/drive/folders/1qL_HtyGcNMztVHEXpAnBRd8kdkxJoM0n?usp=sharing

Malaria detection models are machine learning models designed to identify whether a given cell image is infected with malaria parasites. These models play a crucial role in automating the detection process, assisting medical professionals, and enabling faster diagnosis.

Malaria is a life-threatening disease caused by Plasmodium parasites transmitted through the bites of infected female Anopheles mosquitoes. Microscopic examination of blood smears is the gold standard for malaria diagnosis, but it is time-consuming and requires trained personnel. ML-based malaria detection models offer a potential solution to this problem.

Malaria detection models are typically built using deep learning techniques, with convolutional neural networks (CNNs) being the most commonly used architecture. CNNs excel at analyzing images and can learn intricate patterns and features that aid in distinguishing infected and uninfected cells.

To train a malaria detection model, a dataset of cell images is required. This dataset consists of two classes: infected cells (containing malaria parasites) and uninfected cells. Data augmentation techniques such as rotation, scaling, flipping, and brightness adjustment are often applied to augment the dataset and improve model generalization.

The training process involves feeding the images through the CNN model, adjusting the model's weights and biases based on the predicted outputs, and minimizing a loss function (typically binary cross-entropy) to optimize the model's performance. The model is trained using optimization algorithms such as Adam or Stochastic Gradient Descent (SGD) to find the best set of weights that minimize the loss.

Once trained, the malaria detection model can be used to classify new cell images as infected or uninfected. The model takes an input image, processes it through the layers of the CNN, and produces a probability score indicating the likelihood of infection. A threshold is often applied to convert the probability score into a binary classification (infected or uninfected).

Malaria detection models can be deployed in various ways, such as web applications, mobile apps, or integrated into existing medical systems. These models have the potential to enhance malaria diagnosis by providing quick and accurate results, assisting healthcare professionals in making informed decisions, and potentially enabling remote diagnosis in resource-limited settings.

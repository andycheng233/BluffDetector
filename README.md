# Final Project for CPSC 381

# Deep Buff Detection - CNN + LSTM Facial Expression Classifier

## 1. Problem Statement

Bluffing is a common technique in card games such as poker, where players attempt to deceive their opponents about the strength of their hand to gain an advantage. Detecting whether someone is bluffing is a challenging task, particularly during high-stress situations like competitive card games. These environments are filled with various stimuli, including body language, facial expressions, and other forms of non-verbal communication, which may all contribute to the complexity of the task.

In real-time settings, recognizing these subtle cues can be difficult for humans due to the high cognitive load of processing multiple signals at once. Bluffing detection requires precise observation of facial expressions and other visual indicators. However, many of these cues can be easily faked or manipulated by a skilled player, making it challenging to rely solely on human judgment or traditional behavioral cues.

The objective of this project is to leverage machine learning techniques to analyze visual features that may indicate lying or bluffing. We can achieve a more objective and accurate measure of deception by applying such machine learning techniques. The approach uses real-time facial landmark tracking and analysis to detect variations in facial expressions overtime.

In addition to card games, our model can be used in a variety of real-world applications. For example, in criminal investigations, law enforcement agencies could use these systems to detect deception during suspect interrogations. In political or business negotiations, detecting bluffing can help inform strategic decisions. This technology could even be applied to media, where journalists can assess the credibility of statements made by public figures.

## 2. Run Instructions

1. Download the source files from the zip.

2. Create python virtual environment and install dependencies:

`python -m venv venv
pip install -r requirements.txt`

3. Run the model with the following command:

`python deepBluffDetection.py`

This runs our model4 on our compiled test set, which can be found in:

`truth_test/truth_test_trimmed
lie_test/lie_test_trimmed`

Additionally, model4 was trained on video segments that can be found in:

`truth/truth_trimmed
lie/lie_trimmed`

To make your own videos to train/test the model, follow the instructions:

1. Film segments of a person either lying or telling the truth:

`python record.py`

2. Trim videos to match the 3-second input and create a CSV with paths/labels:

`python convertToCSV.py`

Edit true/false for trainModel to train/evaluate the model with the command:

`python deepBluffDetection.py`

Comments are additionally provided throughout all programs that contain further instructions and clarification.

## 3. Approach

We used a combination of Convolution Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to train a classification model that determines the whether an individual is bluffing or not through video inputs.

CNNs are a great tool for extracting spatial features from video frames and images. Therefore, it is helpful to detecting subtle patterns and variations in video inputs of people bluffing.

LSTMs are a good tool for handing sequential data. This is useful because bluffing or lying happens over a short time period. It is the difference in expression and facial cues over time that allows us to identify whether someone is bluffing or not.

By combining both CNNs and LSTMs, we can take advantage of the CNNs ability to process and extract meaningful features from visual data as well as take advantage of the way LSTM can handle sequences of frames over time.

First, our `convertToCSV.py` file splits longer videos into several three-second clips. Then, we use a CNN model to process each individual frame of the three-second clips. We utilize the Python library MediaPipe to extract a person’s face from the frame to ignore any irrelevant background details and resize the image into a (224,224,3) input. The CNN will extract spatial features of the image as high-dimensional vectors. Next, we will feed in the output from the CNN to the LSTM layer. The LSTM Network will processes a sequence of the features over time to learn temporal patterns in facial expressions to then make a classification prediction.

Feature Extraction Model
```
Input → Conv2D(32,3×3) → ReLU → BatchNorm
      → Conv2D(32,3×3) → ReLU → BatchNorm
      → MaxPool(2×2) → Dropout(0.25)
      → Conv2D(64,3×3) → ReLU → BatchNorm
      → Conv2D(64,3×3) → ReLU → BatchNorm
      → MaxPool(2×2) → Dropout(0.25)
      → Conv2D(128,3×3) → ReLU → BatchNorm
      → Conv2D(128,3×3) → ReLU → BatchNorm
      → MaxPool(2×2) → Dropout(0.25)
      → Flatten
      → Dense(512) → ReLU → BatchNorm → Dropout(0.5)
      → Dense(256) → ReLU
```

The feature-extraction CNN begins by taking a 224×224 RGB image and passing it through three convolutional blocks that learn visual patterns. Each block applies two 3×3 convolutional layers (with 32, then 64, then 128 filters) using ReLU activations and L2 regularization. Then we use batch normalization to speed up and stabilize learning, a 2×2 max-pool to reduce the spatial dimensions, and a 25% dropout to prevent overfitting. Then, the network flattens them into a single vector and feeds it through a 512-unit layer, and projects down to a 256-dimensional feature embedding.

Sequence Classification Model
```
Input → Bidirectional LSTM(128, return_sequences=True)
      → Dropout(0.5)
      → Bidirectional LSTM(64)
      → Dropout(0.5)
      → Dense(32) → ReLU → BatchNorm
      → Dense(1) → Sigmoid
```

The sequence classifier begins by taking in a sequence of 256-dimensional feature vectors per frame and feeding them into a two-layer bidirectional LSTM stack. The first LSTM layer has 128 units per direction and returns a 256-dimensional output at every time step. A 50% dropout prevents overfitting. Then, it’s output passes into a second bidirectional LSTM with 64 units per direction to condenses the sequence into a single 128-dimensional vector. Another 50% dropout is applied to prevent overfitting. This vector is projected through a 32-unit fully layer with ReLU activation and batch normalization. Then, it is reduced to a single logit through a Dense(1) layer and sigmoid activation to return a probability score for binary classification.

## 4. Data

After reviewing different online options, it was hard to find a dataset that would suit our needs. There were very few datasets consisting of video data on bluffing. Many of the datasets focused more on text-based interpretations of lying. Moreover, some of the datasets we identified often had multiple subjects which made it difficult to identify and isolate the individual who was the focus of each data point.

As a result, we decided to collect our own data using the module opencv-python. We recorded segments of people continuously lying or telling the truth and used our script `convertToCSV.py` to split these videos into 3-second clips (and CSV with video paths to these segments and corresponding labels). Using the Python library MediaPipe, we cast a facial mesh over individuals to extract their face from the background frame, which was then inputted in our Convolution Neural Network (CNN) to learn which features are most important in the training processes.

We trained our model using 147 3-second video samples. While this dataset size may appear fairly small, it’s primarily constrained by the computational demands of image processing and the time constraints of this final project. Each 3-second video is recorded at 30 frames per second, resulting in 90 individual frames per input. Each frame, using MediaPipe to extract the face, is resized to (224, 224, 3), yielding a total of approximately 13.5 million pixels processed per video. Both recording the data (and making sure it was clean) and training the model were both very time-consuming. While this amount of data is enough to get started with our model, in order to make it truly reliable and ensure overfitting does not occur we would most likely need to increase our training dataset up to 50-100x.

## 5. Measure of Success

In our original proposal we hoped to achieve around 70% accuracy of our model for determining if an individual is bluffing or not. We understood that bluffing could be a completely subjective action that is unique to each individual. While there are common indicators of lying, not all indicators are present for every individual, and for individuals well trained in suppressing facial cues, it becomes even harder for a human or even a machine model to determine if they are bluffing.

Our original approach using manual feature extraction and logistic regression would result in a very low accuracy rate. Our change to a slightly more complex model using Convolution Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks was meant to increase the amount of features we analyzed and the accuracy of our model’s classification. We currently have a 88% accuracy in our classification which we believe accurately represents the difficulty of the task we are attempting to accomplish. However, we acknowledge that with a more diverse test dataset, our model may not perform as well and that we would need to put in more work to collect data in order to train our model more effectively.

## 6. Results

To evaluate our model, we created a separate test dataset using 52 3-second video segments that were labeled truth or lie. Our results were fairly positive—the model had a 88.46% accuracy. It confidently and correctly identified every person telling the truth (average prediction for probability of person lying was 4.45%) but struggled a bit more with a person telling a lie (average prediction for probability of person lying was 74.60%).

Below are three graphs (Histogram, ROC Curve, and Confusion Matrix) to help demonstrate the accuracy of the model.

Figure 1: Histogram — fairly accurate predictions except for a few misclassifications.
Figure 2: ROC Curve — AUC of 0.96 indicates strong separation.
Figure 3: Confusion Matrix — truth predictions high, lie predictions weaker.

(Insert your images here in Markdown when uploading.)

## 7. Obstacles

At first, our approach was to extract specific facial features that indicated emotional states associated with lying such as blink frequency, variance in mouth movements, variance in face angle, and variance of the face from the facial center. Our Python script `irisDetection.py` used Python libraries OpenCV to capture video and process the frames and MediaPipe to detect facial landmarks that allow us to extract features such as blinking rate and mouth shape. Then, we would use a logistic regression model to predict the probability an individual was lying based on these features.

A difficulty in this implementation was that it was extremely hard to find enough variance in the bluffing versus non-bluffing data. Since the facial features that were being tracked are the same ones that the normal naked eye tends to look for in detecting bluffing, they can easily be faked or manipulated. Common cues such as increased blink frequency, mouth variations, or subtle facial movements are well-known signs of lying but can be consciously controlled by someone trying to bluff. Conversely, individuals telling the truth may also exhibit these behaviors we measure unintentionally.

(Insert Figures 4 and 5 if desired.)

To overcome this limitation, the features that the model would be trained on had to be more accurate and precise, capturing deeper patterns that were not as easily manipulated. Therefore, we shifted to using Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. Our decision for using CNNs was motivated by its usefulness for capturing spatial features from the facial landmarks. Using CNNs will allow us to learn complex patterns in facial expressions without using manual feature extraction, making it not only more efficient but more accurate because we can capture all relevant patterns that was not originally captured in our manual extraction. Using LSTMs help in analyzing sequential patterns over time since bluffing is something that occurs over time rather than in an instance. This combination of CNN and LSTM allows for more accurate deception detection by learning complex patterns that track both temporal and spatial features in our data.

## 8. Reflection

Attempting to determine if someone is lying or not based on facial feature detection and analysis is difficult because of the subjectivity and subtlety of the facial cues. Lying or bluffing is a complex and sometimes unconscious behavior that varies between individuals, which makes it hard to create a one-size-fits-all model. The facial cues of bluffing—such as blink frequency, eye movement, and subtle changes in mouth shape—can be minimal for those who are skilled at masking their expressions. Additionally, these cues may be so subtle that human observers may miss them or misinterpret them. Despite these challenges, by using Convolutional Neural Networks (CNNs), we were able to achieve a more accurate and complex approach to feature extraction. CNNs are capable of learning spatial features automatically from the data, which allowed us to bypass the limitations of manual feature selection. Even though we may not always know which facial features are most indicative of bluffing, CNNs allow the model to learn this autonomously, improving our accuracy without explicitly knowing which features to track.

We think we did a good job recognizing the limitations of our original approach of using manual feature extraction and addressing it with a more complex solution using deep learning techniques. The ability of CNNs to automatically extract relevant features from facial data, paired with the sequence modeling capability of Long Short-Term Memory (LSTM) networks, has provided a much more robust model for detecting bluffing. The model has become more capable of capturing changes in facial expressions over time and learning spatial patterns that are difficult to identify manually.

Moving forward, to further improve the accuracy of our model, we recognize the need for more data cleaning and data augmentation. Since much of our data were collected by ourselves, we recognize the limitation and possible bias of our data collection. We could collect more diverse data and substantially increase the size of our dataset for a more accurate model, which would help with overfitting. Additionally, our data may have a lot of noise and inconsistencies that may affect our model’s performance. If we had more time, we could work on cleaning the data provided in online datasets of video deception, such as finding singular faces in videos to focus on.

Moreover, we can further enhance the feature set by incorporating not only facial features but also full-body feature detection. Human body language can convey different emotional states such as those expressed in lying. Research suggests that people exhibit subtle changes in posture, gestures, and movement when lying. These non-facial cues can complement the facial features we have already extracted to provide a more accurate depiction of a person’s emotional state. By analyzing features such as arm gestures and posture shifts, we can find a more comprehensive set of features to show bluffing.

Furthermore, we could consider integrating voice tone analysis into the system. Voice patterns, such as pitch, rate of speech, and pauses, can give insight into an individual’s intentions when they are speaking. When someone is lying, they might show stress through a higher pitch or speak more quickly. Adding these features could increase the accuracy of our model, as it would be analyzing not just facial and body features, but also vocal signs, creating a multi-modal bluffing detection system.

As we continue to refine our model, we will also need to evaluate its performance across diverse demographics and cultural contexts. Facial expressions of deception may differ between individuals, so ensuring that the model is not too reliant on certain types of behavior is also important to prevent overfitting.

## 9. Conclusion

In this project, we successfully developed a machine learning model capable of detecting bluffing based on facial expressions. Determining if someone is bluffing is a challenging task because of the way someone can control their facial expressions and the way that expressions associated with lying can be changed depending on the individual. However, we were able to use techniques such as Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to improve the robustness of our model.

Our approach involved capturing facial data through video recordings of individuals lying or telling the truth using MediaPipe to extract facial landmarks. By splitting longer videos into three-second clips, we were able to provide the temporal data points relevant for the dynamic nature of lying for our LSTM. We faced many challenges in building this model, such as determining whether to manually extract features or to use a CNN to extract the appropriate features. Moreover, we felt that it was difficult to assemble a diverse and cleaned dataset to train our model that could properly represent real-life examples in the short amount of time that we had, which may have affected the accuracy of our model.

Despite this, our model performed with 88% accuracy. Although we acknowledge that our model was influenced by a variety of factors such as biased data due to a focus on more expressive lying, we are proud of the way that we utilized various machine learning techniques in order to solve this unique problem. Looking ahead, we will look more into expanding our dataset and our features to ensure a more accurate and robust model.

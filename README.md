# 301project
Oral Cancer Detection using Transformer-based Vision (ViT) model

## Project Description: 
•	Goal/Objective: 
In order to improve patient outcomes, oral cancer needs to be diagnosed and treated at an early stage. Using microscopic photographs of mouse tissue, this study aims to provide a deep learning-based solution for identifying oral cancer. Tissue pictures will be classified as normal or malignant using the CNNs model (ResNet50). And then Transformer-based Vision (ViT) model would be applied to the data for possible improvements. (One advantage is that the ViT model can handle images of arbitrary size, whereas CNNs typically require fixed-size input images. Additionally, the ViT model can learn to attend to different parts of the image in a more fine-grained way).The ultimate aim is to produce a system that may be utilised in clinical practise that is both faster and more accurate in detecting oral cancer.

•	Solution Approach: 
Data preprocessing, Transfer learning, Addressing class imbalance, Model evaluation

•	Values: 
In order to improve patient outcomes, oral cancer needs to be diagnosed and treated at an early stage. By using the Transformer-based Vision (ViT) model, tissue pictures will be classified as normal or malignant precisely. The improvement in classification would be of great help to the curing process by the doctors.


## A description of the repository and code structure:
•	Data preprocessing: The dataset will be preprocessed to ensure image quality, normalization, and augmentation. The data augmentation techniques will help to generate more data, which will improve the model's performance.

•	Transfer learning: The model will be fine-tuned on the oral cancer dataset to extract relevant features for classification. The pre-trained model on ImageNet can be used to reduce the number of epochs needed for training, thus saving time and resources.

•	Addressing class imbalance: Techniques such as oversampling and undersampling will be applied to handle the class imbalance problem. Oversampling the minority class will create new synthetic samples, while undersampling the majority class will reduce the number of samples in the majority class.

•	Model evaluation: The model will be evaluated using standard evaluation metrics such as accuracy, precision, recall, and F1-score.


## Example commands to execute the code:
•	Model structure: CNN Tokenizer+ Recurrent Tokenizer + Transformer layer + Fully Connected Classifier

class MyNetwork(nn.Module):

    def __init__(self, transformer, classifier):
        super().__init__()
        self.tokenizer = CNN_Tokenizer(device='cuda', L=16)

        self.recurrent_tokenizer1 = Recurrent_tokenizer(device='cuda', l=16, c=512)
        self.recurrent_tokenizer2 = Recurrent_tokenizer(device='cuda', l=16, c=512)
        self.recurrent_tokenizer3 = Recurrent_tokenizer(device='cuda', l=16, c=512)
        self.recurrent_tokenizer4 = Recurrent_tokenizer(device='cuda', l=16, c=512)

        self.transformer = transformer

        self.classifier = classifier

    def forward(self, x):
        t, features = self.tokenizer(x)

        t = self.recurrent_tokenizer1(features, t)
        t = self.recurrent_tokenizer2(features, t)
        t = self.recurrent_tokenizer3(features, t)
        t = self.recurrent_tokenizer4(features, t)

        t = self.transformer(t)
        op = self.classifier(t)

        return op

model = MyNetwork(transformer, classifier).to('cuda')

•	Data processing：Embedding labels in filename to handle different dataset. Eg. 10001.jpg (Positive), 00001 (Negative).jpg
Shape into (256, 256) and normalization, RGB images are used in this project.

transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

•	Training Process:
used Colab Platform Pro (A100 GPU)
The training process is fast (50 epochs, ~45min)

optimizer_cifar = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()




<img width="367" alt="33" src="https://github.com/SuperStarAdam/301project/assets/98440959/b4336d6f-6cf9-4c49-a65f-db4669b5d88b">



## Results (including charts/tables) and your observations 



<img width="317" alt="截屏2023-05-11 23 40 08" src="https://github.com/SuperStarAdam/301project/assets/98440959/782e1a6c-9ae4-4ee2-ac38-dcdd85012d2d">


<img width="364" alt="截屏2023-05-11 23 40 04" src="https://github.com/SuperStarAdam/301project/assets/98440959/e07eb877-e07a-4a81-a39c-24a698778ab9">




<img width="650" alt="截屏2023-05-11 23 40 19" src="https://github.com/SuperStarAdam/301project/assets/98440959/d011305a-f45c-4b1a-92a0-e91fba8afb75">

The table provides the final results of a cancer detection project, specifically for two types of cancer: oral cancer and skin cancer. The evaluation metrics used to assess the performance of the cancer detection model are true positive rate (TPR), false positive rate (FPR), precision, recall, F-measure, receiver operating characteristic (ROC), and Matthews correlation coefficient (MCC).

For oral cancer, the model achieved a TPR of 0.75, meaning that it correctly identified 75% of the oral cancer cases, and a FPR of 0.25, indicating that 25% of the healthy individuals were falsely identified as having oral cancer. The precision of the model was 0.83, meaning that when the model identified a patient as having oral cancer, it was correct 83% of the time. The recall of the model was 0.75, indicating that it correctly identified 75% of the actual positive cases. The F-measure, which is a harmonic mean of precision and recall, was 0.79. The ROC curve shows the trade-off between sensitivity and specificity, and the area under the curve (AUC) was 0.73. The MCC value of 0.46 indicates a moderate correlation between the predicted and actual oral cancer cases.

For skin cancer, the model achieved a higher TPR of 0.89 and a lower FPR of 0.11, indicating that it correctly identified 89% of the skin cancer cases and only misidentified 11% of healthy individuals as having skin cancer. The precision of the model was 0.86, meaning that when the model identified a patient as having skin cancer, it was correct 86% of the time. The recall of the model was 0.89, indicating that it correctly identified 89% of the actual positive cases. The F-measure was 0.87, which is a good balance between precision and recall. The ROC curve had an AUC of 0.86, indicating a good performance of the model in distinguishing between positive and negative cases. The MCC value of 0.72 suggests a substantial correlation between the predicted and actual skin cancer cases.

Overall, the cancer detection model performed better in detecting skin cancer than oral cancer, as reflected by higher TPR, precision, recall, F-measure, ROC AUC, and MCC values. However, the model's performance for both types of cancer was above average, indicating its potential for use in clinical settings for early detection of cancer. 












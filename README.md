![badge](http://ForTheBadge.com/images/badges/made-with-python.svg) ![badge](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

# covid_cases_predictor
This program is used to develop a LSTM deep learning model that predicts the number of new COVID-19 cases based on previous count. The model was trained using the COVID-19 dataset provided by the Ministry of Health Malaysia.

# How to use
Clone the repository and use the following scripts per your use case:
1. train.py is the script that is used to train the model.
2. categorization_modules.py is the class file that contains the defined functions used in the script for added robustness and reusability of the processes used.
3. The saved model and scaler are available in .h5 and .pkl formats respectively in the 'saved_model' folder.
4. Screenshots of the model architecture, train/test results, true/prediction values plot, and TensorBoard view are available in the 'results' folder.
5. Plot of the training and testing process may be accessed through TensorBoard with the log stored in the 'logs' folder.

# Results
The model developed using 2 hidden LSTM layers was scored using mean absolute percentage error (MAPE), scored at 0.16% error on the test set.

Model architecture:

![model](https://github.com/khaiyuann/covid_cases_predictor/blob/main/results/model.png)

Train/test results (achieved 0.16% MAPE):

![train_results](https://github.com/khaiyuann/covid_cases_predictor/blob/main/results/train_results.png)

True/prediction comparison plot:

![compare_results](https://github.com/khaiyuann/covid_cases_predictor/blob/main/results/compare_results.png)

TensorBoard view:

![tensorboard](https://github.com/khaiyuann/covid_cases_predictor/blob/main/results/tensorboard.png)

# Credits
Thanks to MoH Malaysia (GitHub: MoH-Malaysia) for providing the covid19-public dataset used for the training of the model on GitHub.

Check it out here for detailed information: https://github.com/MoH-Malaysia/covid19-public

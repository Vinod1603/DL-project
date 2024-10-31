# DL-project
we will find the number plate recoginization for every vechile automatically 
we use a video as an input and detect the number plates.
we use the Yolov8 pre-trained model  was used to detect vehicles.
A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using a data set


#project setup
Make an environment with python=3.10 using the following command
conda create --prefix ./env python==3.10 -y
Activate the environment
source activate ./env
Install the project dependencies using the following command
pip install -r requirements.txt
Run main.py with the sample video file to generate the test.csv file
python main.py
Run the add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
python add_missing_data.py
Finally run the visualize.py passing in the interpolated csv files and hence obtaining a smooth output for license plate detection.
python visualize.py

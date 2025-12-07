The Extra SMIle - Project Group 05.10


PROJECT OVERVIEW

The objective of our web application is to help investors in the Swiss Market Index (SMI) to make data-driven investment decisions. The application goes beyond simple price charts and analyses the risk-return dynamics of the SMI-assets and the correlation of the different assets. It furthermore integrates Machine Learing by volatility forecasting to help investors gain model-based insights about the future behavior of assets.


FILE STRUCTURE

The_Extra_SMIle.py: The main file containing all our code.

requirements.txt: A list of Python libraries required to run this application.

Doumentation_Project_Group_05.10.pdf: A file including the names of our team members, our contribution matrix and other documentations.

README.txt: This instruction file.


RUN THE APPLICATION

Ensure you have Python installed.
Unzip the project folder.
Open your terminal or command prompt in this folder.

Install the required dependencies by running the following command in your terminal:
pip install -r requirements.txt

Launch the application:
streamlit run The_Extra_SMIle.py

NOTE ON DATA LOADING

The application fetches live data from Yahoo Finance. Please ensure you have an active internet connection when running the app. The initial data load might take a few seconds depending on your connection speed.

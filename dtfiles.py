import os
import glob
import pandas as pd

path = "C:/Users/mauri/Documents/Zoom/Econ 262 1003_Attendance/PollReport"
all_files = glob.glob(path + "/*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, header=0)
    li.append(df)

drf = pd.concat(li,  ignore_index=True)
drf.drop(columns=['Answer'], inplace=True)
drf.rename(columns={"#": "User_Name", "User Name": "User_Email",
                    "User Email": "Question", "Question": "Answer"}, inplace=True)
os.chdir('C:/Users/mauri/Documents/Zoom/Econ 262 1003_Attendance/Dta Files')
drf.to_csv('PollReport_1003.csv', index=False)


path = "C:/Users/mauri/Documents/Zoom/Econ 262 1004_Attendance/PollReport"
all_files = glob.glob(path + "/*.csv")

li = []
for filename in all_files:
    df = pd.read_csv(filename, header=0)
    li.append(df)

drf2 = pd.concat(li,  ignore_index=True)
print(drf2.columns)
drf2.drop(columns=['Answer'], inplace=True)
drf2.rename(columns={"#": "User_Name", "User Name": "User_Email",
                     "User Email": "Question", "Question": "Answer"}, inplace=True)
os.chdir('C:/Users/mauri/Documents/Zoom/Econ 262 1003_Attendance/Dta Files')
drf.to_csv('PollReport_1004.csv', index=False)

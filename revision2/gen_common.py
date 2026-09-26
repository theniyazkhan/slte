# Exact reproduction of the ORIGINAL generation preprocessing (Initial_Experiments.ipynb cell 6)
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
TARGET = 'No-show'
DISCRETE = ['Gender','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap','SMS_received',
            'ScheduledDayOfWeek','AppointmentDayOfWeek','ScheduledMonth', TARGET]
def load_gen_train():
    df = pd.read_csv('MedicalAppointment.csv')
    df.drop(columns=['PatientId','AppointmentID'], inplace=True)
    df = df[df['Age'] >= 0].copy()
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], utc=True)
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], utc=True)
    df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days.clip(lower=0)
    df['ScheduledDayOfWeek'] = df['ScheduledDay'].dt.dayofweek
    df['AppointmentDayOfWeek'] = df['AppointmentDay'].dt.dayofweek
    df['ScheduledMonth'] = df['ScheduledDay'].dt.month
    df.drop(columns=['ScheduledDay','AppointmentDay'], inplace=True)
    df['Gender'] = (df['Gender'] == 'M').astype(int)
    fm = df['Neighbourhood'].value_counts(normalize=True).to_dict()
    df['Neighbourhood_freq'] = df['Neighbourhood'].map(fm); df.drop(columns=['Neighbourhood'], inplace=True)
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)
    X = df.drop(columns=[TARGET]); y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return pd.concat([Xtr, ytr], axis=1)
def postprocess(s):
    s = s.copy()
    s['Age'] = s['Age'].clip(0,115).round().astype(int)
    s['WaitDays'] = s['WaitDays'].clip(0).round().astype(int)
    for c in ['Gender','Scholarship','Hipertension','Diabetes','Alcoholism','SMS_received',TARGET]:
        s[c] = s[c].round().clip(0,1).astype(int)
    s['Handcap'] = s['Handcap'].round().clip(0,4).astype(int)
    return s

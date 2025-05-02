import pandas as pd
import numpy as np
import random

blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
genders = ['Male', 'Female', 'Other']

def generate_sample():
    age = np.random.randint(1, 90)
    bloodGroup = random.choice(blood_groups)
    gender = random.choice(genders)
    height = np.random.normal(165, 10)
    weight = np.random.normal(70, 15)
    
    hasDiabetes = random.choice([0, 1])
    hasBpHigh = random.choice([0, 1])
    hasBpLow = random.choice([0, 1])

    heartRate = np.random.normal(80, 15)
    SpO2 = np.random.normal(97, 2)
    temperature = np.random.normal(36.8, 0.5)

    height = np.clip(height, 140, 200)
    weight = np.clip(weight, 40, 150)
    heartRate = np.clip(heartRate, 30, 180)
    SpO2 = np.clip(SpO2, 70, 100)
    temperature = np.clip(temperature, 34, 41)

    abnormal = 0

    if hasDiabetes:
        if not (60 <= heartRate <= 110):
            abnormal = 1
        if SpO2 < 93:
            abnormal = 1
        if not (36.1 <= temperature <= 37.5):
            abnormal = 1

    elif hasBpHigh:
        if not (60 <= heartRate <= 100):
            abnormal = 1
        if SpO2 < 95:
            abnormal = 1
        if not (36.1 <= temperature <= 37.2):
            abnormal = 1

    elif hasBpLow:
        if not (55 <= heartRate <= 100):
            abnormal = 1
        if SpO2 < 95:
            abnormal = 1
        if not (36.1 <= temperature <= 37.2):
            abnormal = 1

    else:
        if not (60 <= heartRate <= 100):
            abnormal = 1
        if SpO2 < 95:
            abnormal = 1
        if not (36.1 <= temperature <= 37.2):
            abnormal = 1

    return {
        'age': age,
        'bloodGroup': bloodGroup,
        'hasBpHigh': hasBpHigh,
        'hasBpLow': hasBpLow,
        'gender': gender,
        'height': height,
        'hasDiabetes': hasDiabetes,
        'weight': weight,
        'heartRate': heartRate,
        'SpO2': SpO2,
        'temperature': temperature,
        'label': 'abnormal' if abnormal else 'normal'
    }

n_samples = 5000
data = [generate_sample() for _ in range(n_samples)]
df = pd.DataFrame(data)

df.to_csv('synthetic_vitals_dataset_conditioned.csv', index=False)

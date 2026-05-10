# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:58:57 2026

@author: Administrator
"""

import pandas as pd
import numpy as np
import os

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# Number of samples per severity class
n_per_class = 200   # total 600 rows
np.random.seed(42)  # for reproducibility

def generate_samples(severity, n):
    # Generate features that are realistic for each severity
    if severity == 'Critical':
        age = np.random.randint(40, 90, n)
        heart_rate = np.random.randint(100, 140, n)
        breathing_rate = np.random.randint(25, 40, n)
        consciousness = np.random.choice([0,1,2], n, p=[0.1, 0.3, 0.6])
        injury_severity = np.random.choice([1,2,3,4,5], n, p=[0.05, 0.1, 0.2, 0.3, 0.35])
        arrival_mode = np.random.choice(['ambulance','walk','helicopter'], n, p=[0.8, 0.1, 0.1])
    elif severity == 'Moderate':
        age = np.random.randint(20, 80, n)
        heart_rate = np.random.randint(80, 115, n)
        breathing_rate = np.random.randint(18, 30, n)
        consciousness = np.random.choice([0,1,2], n, p=[0.4, 0.4, 0.2])
        injury_severity = np.random.choice([1,2,3,4,5], n, p=[0.1, 0.2, 0.3, 0.2, 0.2])
        arrival_mode = np.random.choice(['ambulance','walk','helicopter'], n, p=[0.6, 0.2, 0.2])
    else:  # Minor
        age = np.random.randint(15, 60, n)
        heart_rate = np.random.randint(60, 100, n)
        breathing_rate = np.random.randint(12, 25, n)
        consciousness = np.random.choice([0,1,2], n, p=[0.7, 0.25, 0.05])
        injury_severity = np.random.choice([1,2,3,4,5], n, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        arrival_mode = np.random.choice(['ambulance','walk','helicopter'], n, p=[0.5, 0.3, 0.2])
    
    return pd.DataFrame({
        'age': age,
        'heart_rate': heart_rate,
        'breathing_rate': breathing_rate,
        'consciousness': consciousness,
        'injury_severity': injury_severity,
        'arrival_mode': arrival_mode,
        'triage': [severity] * n
    })

# Generate balanced data
df_crit = generate_samples('Critical', n_per_class)
df_mod = generate_samples('Moderate', n_per_class)
df_min = generate_samples('Minor', n_per_class)

df_balanced = pd.concat([df_crit, df_mod, df_min], ignore_index=True)
# Shuffle rows
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to the expected location
df_balanced.to_csv("dataset/synthetic_medical_triage.csv", index=False)
print(f"✅ Balanced dataset saved: {len(df_balanced)} rows")
print(df_balanced['triage'].value_counts())
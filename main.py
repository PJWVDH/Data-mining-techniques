import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def plot_data(data):
    df = pd.DataFrame(data[1:], columns=data[0])
    df['Value'] = pd.to_numeric(df['Value'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Value'], marker='o')
    plt.title('Data Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'dataset_moodPsmartphone.csv'  # Replace with your CSV file path
    data = read_csv(file_path)
    plot_data(data)

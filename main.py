import csv
import matplotlib
matplotlib.use('Agg')
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

    # parse time and numeric values
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # drop any rows that failed conversion
    df = df.dropna(subset=['time', 'value']).sort_values('time')

    # missing values are filled with the mean of the column
    df['value'].fillna(df['value'].mean(), inplace=True)

    # rolling average for smoother trend
    df['rolling'] = df['value'].rolling(window=7, min_periods=1).mean()

    # initial timeseries plot with points
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True, sharex=False)

    ax[0].plot(df['time'], df['value'], marker='o', linestyle='-', label='Mood', alpha=0.7)
    ax[0].plot(df['time'], df['rolling'], color='red', linestyle='--', linewidth=2, label='7-point rolling mean')
    ax[0].set_title('Mood over Time with Rolling Mean')
    ax[0].set_xlabel('Timestamp')
    ax[0].set_ylabel('Mood Value')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # distribution boxplot
    ax[1].boxplot(df['value'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax[1].set_title('Mood Value Distribution')
    ax[1].set_xlabel('Mood Value')
    ax[1].set_yticks([])

    output_path = 'mood_plot.png'
    plt.savefig(output_path, dpi=150)
    print(f'Saved plot to {output_path}')

if __name__ == "__main__":
    file_path = 'dataset_mood_smartphone.csv'
    data = read_csv(file_path)
    plot_data(data)




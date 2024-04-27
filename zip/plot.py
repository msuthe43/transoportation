import pandas as pd
import matplotlib.pyplot as plt

def plot_intersection_performance(csv_file):
    # Load the CSV data into a DataFrame
    data = pd.read_csv(csv_file)

    # Assume there's a column that represents each step or time; if not, consider the index as the time
    if 'step' not in data.columns:
        data['step'] = data.index

    # Calculate rolling averages to smooth the data
    window_size = 1100  # Adjust this based on your data for desired smoothing
    data['rolling_total_waiting_time'] = data['system_total_waiting_time'].rolling(window=window_size).mean()
    data['rolling_mean_waiting_time'] = data['system_mean_waiting_time'].rolling(window=window_size).mean()
    data['rolling_mean_speed'] = data['system_mean_speed'].rolling(window=window_size).mean()

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Total system waiting time
    axs[0].plot(data['step'], data['rolling_total_waiting_time'], label='Rolling Avg Total Waiting Time', color='red')
    axs[0].set_ylabel('Total Waiting Time')
    axs[0].set_title('Total System Waiting Time Over Time')
    axs[0].legend()

    # System mean waiting time
    axs[1].plot(data['step'], data['rolling_mean_waiting_time'], label='Rolling Avg Mean Waiting Time', color='blue')
    axs[1].set_ylabel('Mean Waiting Time')
    axs[1].set_title('System Mean Waiting Time Over Time')
    axs[1].legend()

    # System mean speed
    axs[2].plot(data['step'], data['rolling_mean_speed'], label='Rolling Avg Mean Speed', color='green')
    axs[2].set_ylabel('Mean Speed (m/s)')
    axs[2].set_title('System Mean Speed Over Time')
    axs[2].legend()

    # Set common labels
    plt.xlabel('Simulation Step')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Specify the path to your CSV file
    csv_file_path = r'C:\Users\saver\3010project\out_conn0_ep1.csv'
    plot_intersection_performance(csv_file_path)

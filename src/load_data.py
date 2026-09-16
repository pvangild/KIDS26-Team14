import pymatreader

def load_mat_file(filepath):
    data = pymatreader.read_mat(filepath)
    return data

if __name__ == "__main__":
    filepath = "data/raw/ahiData-pats-800003-baseline.mat"  # replace with your actual filename
    data = load_mat_file(filepath)
    print(data.keys())
    print("Metrics:", data['metrics'])
    print("Signal channels:", data['signals']['labels'])
    print("Signal data shape:", data['signals']['data'].shape)
    print("Events:", data['events'])
    apneas = data['events']['apneas']
    hypopneas = data['events']['hypopneas']
    print("Number of apnea events:", len(apneas))
    print("Number of hypopnea events:", len(hypopneas))
    print("First apnea event (start, end):", apneas[0])
    import matplotlib.pyplot as plt

    start, end = apneas[0]
    window = slice(start - 100, end + 100)  # a little padding before/after
    time = data['signals']['time'][window]
    airflow = data['signals']['data'][window, 1]  # column 1 = airFlow
    spo2 = data['signals']['data'][window, 2]     # column 2 = SpO2

    fig, ax1 = plt.subplots()
    ax1.plot(time, airflow, color='blue', label='Airflow')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Airflow', color='blue')

    ax2 = ax1.twinx()
    ax2.plot(time, spo2, color='red', label='SpO2')
    ax2.set_ylabel('SpO2', color='red')

    plt.title('Airflow and SpO2 around first apnea event')
    plt.show()
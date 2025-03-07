import os
import scipy.io
import matplotlib.pyplot as plt

def visualize_mat_file(file_path,cartella):
    data = scipy.io.loadmat(file_path)
    if 'feats' in data:
        feats_data = data['feats']
        num_points = feats_data.size
        data_shape = feats_data.shape
        # print(f"Data from feats: {feats_data}")
        # print(f"Number of points: {num_points}")
        # print(f"Data shape: {data_shape}")
        
        # print(f"Data tempi vecchi: {data['org_sample_rate']}")
        # print(f"Data tempi nuovi: {data['curr_sample_rate']}")
        # print(f"Data tempi totali vecchi: {data['org_sample_size']}")
        # print(f"Data tempi totali nuovi: {data['curr_sample_size']}")
        # print(data.keys())
        
        for key, value in data.items():
            if key in ['__header__', '__version__', '__globals__']:
                continue 
            if key in ['idx', 'mean', 'std']:
                continue 
            
            print(key, value.shape)
            
            if key == 'segment_i':
                print('il valore di segment è ', value)
            
        
        
        # plt.figure()
        
        
        # plt.plot(feats_data[0, :])
        # plt.title("Data from feats [0, :]")
        # file = f'feats_plot{cartella}.png'
        # plot_save_path = os.path.join('/home/sbartucci/ECG_forecasting', file)
        # plt.savefig(plot_save_path)
        # plt.close()
        output_dir = '/home/sbartucci/ECG_forecasting'
        output_path = os.path.join(output_dir, 'feats_data_plots.png')

        titles = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))

        for i in range(12):
            row = i // 4
            col = i % 4
            axes[row, col].plot(feats_data[i, :])
            axes[row, col].set_title(f'Plot of feats_data[{titles[i]}]')
            axes[row, col].set_xlabel('Index')
            axes[row, col].set_ylabel('Value')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close() 

root = os.getcwd()

cartelle = ['segmented']

for cartella in cartelle:
    mat_files_dir = os.path.join(root, f'Test_on_physionet/processed_root/physionet2021/{cartella}')
    print(cartella)

    for file_name in os.listdir(mat_files_dir):
        if file_name.endswith('.mat'):
            file_path = os.path.join(mat_files_dir, file_name)
            
            visualize_mat_file(file_path,cartella)
            print()     
            
            break  # Stampa solo il primo file .mat trovato



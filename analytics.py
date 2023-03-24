from helper_functions import load_model, unpack_pickle
import image_metrics as im
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

import matplotlib.pyplot as plt
# Need to load models
# Need pickled file as well as model archs
# Model archs cannot be imported form notebook easily going to store them in # model_arch.py

# Get Data Ready 

path = "./compressed_images/cifar-data/"
train_file_path = path + "train_32_16_4_8.dat"
valid_file_path = path + "valid_32_16_4_8.dat"
test_file_path = path + "test_32_16_4_8.dat"
x_train, y_train = unpack_pickle(train_file_path)
x_valid, y_valid = unpack_pickle(valid_file_path)
x_test, y_test = unpack_pickle(test_file_path)

def prepare_data_for_dataset(x, y):
    x = np.stack( x, axis=0)
    x = x.swapaxes(1, 3)
    y = np.stack( y, axis=0)
    y = y.swapaxes(1, 3)
    return x, y

x_train, y_train = prepare_data_for_dataset(x_train, y_train)
print("x_train new dimensions:", x_train.shape)
print("y_train new dimensions:", y_train.shape)

x_valid, y_valid = prepare_data_for_dataset(x_valid, y_valid)
print("x_valid new dimensions:", x_valid.shape)
print("y_valid new dimensions:", y_valid.shape)

x_test, y_test = prepare_data_for_dataset(x_test, y_test)
print("x_test new dimensions:", x_valid.shape)
print("y_test new dimensions:", y_valid.shape)

class CompressionDataset(Dataset):
    def __init__(self, compressed_images, original_images, transforms=None) -> None:
        
        # scale input from 0 to 1
        self.original = original_images/255
        self.compressed = compressed_images/255
        
        
        if transforms != None: 
            for transform in transforms:
                self.original = transform(self.original)
                self.compressed = transform(self.compressed)
        super().__init__()
    def __len__(self):
        return len(self.original)
    def __getitem__(self, index):
        original_image = self.original[index]
        compressed_image = self.compressed[index]
        sample = {"Compressed":compressed_image, "Original":original_image}
        return sample


def ToTensor(array):
    return torch.from_numpy(array).float()


train_data_set = CompressionDataset(x_train, y_train, transforms= [ToTensor])
valid_data_set = CompressionDataset(x_valid, y_valid, transforms=[ToTensor])
test_data_set = CompressionDataset(x_test, y_test, transforms=[ToTensor])

batch_size = 5
train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
valid_data_loader = DataLoader(valid_data_set, batch_size=batch_size, shuffle=False)
test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)


from model_archs import *

#### ENTER THE MODELS YOU WANT TO ANALYZE HERE! 

model_dict = {
    "Dense" : load_model("./trained_models/dense_model", Dense_Connection_Model),
    "Residual" : load_model("./trained_models/residual_model", Residual_Model),
    "SRGAN" : load_model("./trained_models/G-Andrew-3-24", SR_Generator),
    "Ex SRGAN" : load_model("./trained_models/G_SR_Alex_Trenton_3_23", Ex_SR_Generator )
}



def perform_analytics(model_dict, num_batches, data_loader, **other_metrics):
    # Other metrics supports "time" with time data and "model_size" being True or False
    benchmark = im.Benchmark_Suite()

    # Get metrics for each model
    model_data = {}
    psnrb_data = []
    ssim_data = []
    times = []

    for model_num, model_name in enumerate(model_dict):
        model = model_dict[model_name]
        if other_metrics.get("loss") == None:
            print("Loss not specified, defaulting to MSELoss")
            loss_func = nn.MSELoss()
        else:
            loss_func = other_metrics["loss"][model_num]
        print("MODEL NAME: ", model_name)
        losses, metrics, time = benchmark.dataset_performance(network=model, loss_func=loss_func, data_loader=data_loader, stop=num_batches, average=True)
        #losses_2, metrics_2, time_2= benchmark.dataset_performance(network=model, loss_func=loss_func, data_loader=data_loader, stop=num_batches, average=True)
    
        model_data[model_name] = [losses, metrics] # Store for use later
        psnrb_data.append(metrics[0]["psnr"]) # Metrics are averaged, so there is only one dict in our list of metrics
        ssim_data.append(metrics[0]["ssim"])
        times.append(time)


    # Now plot these suckers, want to plot based on two metrics of interest and any metrics passed in 

    # First the two metrics we are interested in

    model_names_list = [model_name for model_name in model_data]
    figure_num = 1

    plt.figure(figure_num)
    figure_num +=1
    plt.title("PSNR")
    plt.legend(model_names_list) # Legend is the model names
    plt.bar(model_names_list, psnrb_data)
    plt.show()
    plt.figure(figure_num)
    figure_num +=1
    plt.title("SSIM")
    plt.legend(model_names_list) # Legend is the model names
    plt.bar(model_names_list, ssim_data)
    plt.show()
    plt.figure(figure_num)
    figure_num +=1
    plt.title("Decompression Time")
    plt.legend(model_names_list) # Legend is the model names
    plt.bar(model_names_list, times)
    plt.show()
    
    
    # Then do the same thing for each keyword metric
    if other_metrics.get("model_size") == True:
        # Get model sizes
        total_params_list = []
        for model in model_dict.values():
            total_params = 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                params = parameter.numel()
                total_params+=params
            total_params_list.append(total_params)

        plt.figure(figure_num)
        figure_num +=1
        plt.title("Model Size")
        plt.ylabel("Num Trainable Parameters")
        plt.legend(model_names_list) # Legend is the model names
        plt.bar(model_names_list, total_params_list)
        plt.show()
        pass


# Plotting Code for Each Model 

def model_plots(model_dict, num_images, data_loader):
    num_models = len(model_dict) # Number of rows
    # Going to have a input image, y hat then y
    fig, axs = plt.subplots(num_images, num_models + 2)
    # Going to plot as we go! 
    for model_column, model_name in enumerate(model_dict):
        model = model_dict[model_name]
        model.eval()
        for idx, batch in enumerate(data_loader):

            x = batch["Compressed"]
            y = batch["Original"]
            y_hat = model(x)

            # Now plot up to downm, left to right(for each iteration)
            for image_index in range(x.shape[0]):
                axs[image_index, 0].imshow(x[image_index].detach().numpy().swapaxes(0, 2))
                img = y_hat[image_index].detach().numpy().swapaxes(0, 2)
                model_image = img = ((img / img.max()) * 255).astype(np.uint8)
                axs[image_index, model_column+1].imshow(img)
                axs[image_index, num_models+1].imshow(y[image_index].detach().numpy().swapaxes(0, 2))
            
                # Formatting
                axs[image_index, 0].set_xticks([])
                axs[image_index, 0].set_yticks([])
                axs[image_index, 0].set_title("x")
                
                    
                axs[image_index, model_column+1].set_xticks([])
                axs[image_index, model_column+1].set_yticks([])
                if image_index == 0:
                    axs[image_index, model_column+1].set_title(f"{model_name}\ny_hat")
                else:
                    axs[image_index, model_column+1].set_title("y_hat")



                axs[image_index, num_models+1].set_xticks([])
                axs[image_index, num_models+1].set_yticks([])
                axs[image_index, num_models+1].set_title("y")

                if image_index == num_images-1:
                    break
            break
    fig.tight_layout()
    plt.show()



perform_analytics(model_dict, 3, test_data_loader, model_size = True)

#model_plots(model_dict, 5, test_data_loader)
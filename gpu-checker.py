import torch
def gpu_checker():
    return (torch.cuda.device_count(), torch.cuda.current_device(),
    torch.cuda.device(0), torch.cuda.get_device_name(0),
    torch.cuda.is_available())


print('Devices Connected: ', gpu_checker()[0])
print('Device In Use: ', gpu_checker()[1])
print('Device Location: ', gpu_checker()[2])
print('Device Name: ', gpu_checker()[3])
print('Is Device Available: ', gpu_checker()[4])

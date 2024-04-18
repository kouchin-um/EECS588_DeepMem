from train_model_py3 import *
from test_model_py3 import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

OUTPUT_PATH = '../create_memory_graphs/graph/'
input_size = 64
hidden_size = 128
model_path = 'model_torch_d.0'

def load_data_G(model, target_obj_type='_ETHREAD'):
    log('Object type: ' + target_obj_type)
    
    # load the threshold of the target object type
    obj_type_threshold = load_threshold(target_obj_type)

    dict_key_node_to_weight = load_key_node_weight(target_obj_type)
    log(str(dict_key_node_to_weight))
    output_vector_size = get_outout_vector_size(dict_key_node_to_weight)
    log('output_vector_size:\t%d' %output_vector_size)

    _, _, list_file_test = get_file_list(target_obj_type)
    log(list_file_test)
    i = 0
    # for file_graph in list_file_train:
    #     log(file_graph)
    #     list_vector, _, _, _, _, _, _, _ = read_dataset(file_graph, output_vector_size, dict_key_node_to_weight, target_obj_type)
    #     list_vector = torch.tensor(list_vector)
    #     outputs = model(list_vector.float())
        
    #     with open(file_graph, 'r') as f:
    #         lines = [x for x in f]

    #     for idx, line in enumerate(lines):
    #         s = line.strip().split('\t')
    #         obj_type = None
    #         if len(s) >= 8:
    #             obj_type = s[7].split('@')[0]
    #         if obj_type == target_obj_type:
    #             n_bit = int(s[-3])
    #             for idx_bit in range(n_bit, 32):
    #                 outputs[idx][idx_bit] = 0
    #         else:
    #             outputs[idx] = list_vector[idx]

    #     outputs = outputs.detach().numpy().astype(np.uint8)
    #     np.save(f'g_model/output.{i}.npy', outputs)
    #     i += 1
    for file_graph in list_file_test:
        log(file_graph)
        list_vector, _, _, _, _, _, _, _ = read_dataset(file_graph, output_vector_size, dict_key_node_to_weight, target_obj_type)
        list_vector = torch.tensor(list_vector)
        outputs = model(list_vector.float())
        
        with open(file_graph, 'r') as f:
            lines = [x for x in f]

        for idx, line in enumerate(lines):
            s = line.strip().split('\t')
            obj_type = None
            if len(s) >= 8:
                obj_type = s[7].split('@')[0]
            if obj_type == target_obj_type:
                n_bit = int(s[-3])
                for idx_bit in range(n_bit, 32):
                    outputs[idx][idx_bit] = 0
            else:
                outputs[idx] = list_vector[idx]

        outputs = outputs.detach().numpy().astype(np.uint8)
        np.save(f'g_model/output.{i}.npy', outputs)
        i += 1
    
    return outputs

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(x) * 255
        return x.round()
    
def log(message):
    print('%s\t%s' %(strftime("%Y-%m-%d %H:%M:%S", gmtime()), message))
    sys.stdout.flush()


def loss_G(inputs, model_path):
    model = torch.load(model_path)
    model_output = model(inputs)
    model_output = torch.clamp(model_output, min=epsilon, max=1-epsilon)
    loss = -torch.log(model_output)
    return loss

def main():
    model = Generator()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #torch.save(model.state_dict(), 'g_model/model.0')
    num_epochs = 10
    for epoch in range(num_epochs):
        inputs = load_data_G(model)
        loss = loss_G(inputs, model_path)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    #torch.save(model.state_dict(), 'g_model/model.0')

if __name__ == '__main__':
    main()

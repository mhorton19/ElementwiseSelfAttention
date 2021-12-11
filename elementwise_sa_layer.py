import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_softmax import masked_softmax

class ElemetwiseSaSummarizer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ElemetwiseSaSummarizer, self).__init__()
        self.predict_global = nn.Linear(input_size, output_size, bias=False)
        self.predict_weight = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x, attention_mask=None):

        global_vecs = self.predict_global(x).permute(0, 2, 1)

        weight = self.predict_weight(x).permute(0, 2, 1)
        #print(weight.size())
        #print(attention_mask.size())
        if(attention_mask != None):
            weight_vec = masked_softmax(weight, attention_mask.unsqueeze(1), dim=-1)
        else:
            weight_vec = torch.softmax(weight, dim=-1)

        #print(weight_vec.size())
        #print(global_vecs.size())


        global_vec = torch.matmul(global_vecs.unsqueeze(-2), weight_vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)

        return global_vec

class ElementwiseSaLayer(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super(ElementwiseSaLayer, self).__init__()

        self.intermediate_size = intermediate_size

        self.predict_global = nn.Linear(input_size, intermediate_size, bias=False)
        self.predict_weight = nn.Linear(input_size, intermediate_size, bias=False)
        self.predict_gate = nn.Linear(input_size, intermediate_size)
        self.predict_condense = nn.Linear(intermediate_size, input_size, bias=False)
        self.global_dropout = nn.Dropout(p=0.1, inplace=True)
        self.elementwise_dropout = nn.Dropout(p=0.1, inplace=True)

    def forward(self, x, attention_mask=None):

        global_vecs = self.predict_global(x).permute(0, 2, 1)

        weight = self.predict_weight(x).permute(0, 2, 1)
        #print(weight.size())
        #print(attention_mask.size())
        if(attention_mask != None):
            weight_vec = masked_softmax(weight, attention_mask.unsqueeze(1), dim=-1)
        else:
            weight_vec = torch.softmax(weight, dim=-1)

        #print(weight_vec.size())
        #print(global_vecs.size())


        global_vec = torch.matmul(global_vecs.unsqueeze(-2), weight_vec.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        gates = torch.sigmoid(self.predict_gate(x))
        global_vec = self.global_dropout(global_vec)
        #gates = self.predict_gate(x)
        gated = global_vec.unsqueeze(1) * gates
        out = self.predict_condense(gated)

        #print(global_vec.size())
        return self.elementwise_dropout(out)

class FCLayer(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super(FCLayer, self).__init__()

        self.norm = nn.LayerNorm(intermediate_size)
        self.fc1 = nn.Linear(input_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, input_size, bias=False)
        self.dropout = nn.Dropout(p=0.1, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.norm(x))
        x = self.fc2(x)

        return self.dropout(x)

class PrenormElementwiseUnit(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super(PrenormElementwiseUnit, self).__init__()
        self.esa = ElementwiseSaLayer(input_size, intermediate_size)
        self.fc = FCLayer(input_size, intermediate_size)
        self.norm_esa = nn.LayerNorm(input_size)
        self.norm_fc = nn.LayerNorm(input_size)

    def forward(self, x, attention_mask=None):
        x = x + self.esa(self.norm_esa(x), attention_mask=attention_mask)
        x = x + self.fc(self.norm_fc(x))
        return x


class LSTM_Baseline_Unit(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super(LSTM_Baseline_Unit, self).__init__()
        self.esa = nn.LSTM(input_size, intermediate_size, proj_size=intermediate_size//2, bidirectional=True, batch_first=True)
        #self.fc = FCLayer(input_size, intermediate_size)
        self.norm_esa = nn.LayerNorm(input_size)

    def forward(self, x, attention_mask=None):
        x = x + self.esa(self.norm_esa(x))[0]

        return x

class LSTM_Baseline_Summarization_Unit(nn.Module):
    def __init__(self, input_size, intermediate_size):
        super(LSTM_Baseline_Summarization_Unit, self).__init__()
        self.esa = nn.LSTM(input_size, intermediate_size, proj_size=intermediate_size//2, bidirectional=True, batch_first=True)
        # self.fc = FCLayer(input_size, intermediate_size)
        self.norm_esa = nn.LayerNorm(input_size)

    def forward(self, x, attention_mask=None):
        x = x + self.esa(self.norm_esa(x))[0]

        return x[:, -1, :]

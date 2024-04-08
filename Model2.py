from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pandas as pd
import torch
import attention

import numpy as np
from torch.nn import init

class BertForSequenceClassification(nn.Module):

    def __init__(self, config,num_labels):
        super(BertForSequenceClassification, self).__init__()

        # hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(0.1)
        self.softmax=nn.Softmax(dim=1)
        self.sigmond=nn.Sigmoid()
        self.fake_classifier = nn.Linear(config.hidden_size, num_labels) #768->34
        self.classifier_gate = nn.Linear(config.hidden_size, 2*num_labels) #768->34
        self.classifier = nn.Linear(config.hidden_size+num_labels , num_labels) #68->34
        self.classifierboth = nn.Linear(config.hidden_size + 2*num_labels, num_labels)  # 68->34
        self.classifier_sam = nn.Linear(config.hidden_size, num_labels) #768->34
        self.jiang = nn.Linear(num_labels, num_labels)  # 3*34->34

        # self.classifier2 = nn.Linear(2*config.hidden_size, config.num_labels)
        #
        # print(config.hidden_size)
        # self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        shangyiceng_output = None ,#shangyiceng_output Shape:[Batch,config.num_labels] tensor
        pooled_output1=None
    ):
        # pooled_output1 = self.dropout(pooled_output)
        if shangyiceng_output != None:
            length=int(len(shangyiceng_output[0])/2)
            # print(length)
            # print('shangyiceng_output',shangyiceng_output)
            fake_predict = self.fake_classifier(pooled_output1) #边策改，假的输出
            # print(fake_predict.shape)\

            gate_predict1 = self.sigmond(self.fake_classifier(pooled_output1))
            gate_out_put1 = gate_predict1.mul(shangyiceng_output[:,:length])


            gate_predict2 = self.sigmond( self.fake_classifier(pooled_output1))
            gate_out_put2 = gate_predict2.mul(shangyiceng_output[:,length:])

            # a=fake_predict
            #
            # b=shangyiceng_output[:,:length]
            # c = shangyiceng_output[:,length:]
            # input=torch.stack([fake_predict, gate_out_put1, gate_out_put2], dim=0)
            # sa = attention.ScaledDotProductAttention(d_model=length, d_k=length, d_v=length, h=8)
            # input_classifiar_tensor = sa(input, input, input)
            # logits = self.jiang(input_classifiar_tensor[0] + input_classifiar_tensor[1] + input_classifiar_tensor[2])


            logits = self.jiang(fake_predict+gate_out_put1+gate_out_put2)

            # print(input_classifiar_tensor.shape)
            # input_classifiar_tensor = torch.add(input=fake_predict, alpha=1, other=gate_out_put)


            # logits = self.jiang(input_classifiar_tensor.permute(1, 2, 0))
            # d=[  1., 30,184.1,166.3,45.2,271.3,245.5,271.3,139.35,234.36,17.9,
            #      166.3,47.3, 147.3, 286.4,  55.4,396.6, 139.3,  95.5, 234.4]
            d = [1 for i in range(20)]
            # d=[]
        else:
            logits = self.classifier_sam(pooled_output1) #边策改，假的输出
            # print(logits.shape)
            # d = [1., 1.92, 21., 40., 20., 20., 37., 69*4., 44*4., 44*3., 75.]


            d=[1 for i in range(11)]


        # weight = torch.FloatTensor(d)
        loss_fct = CrossEntropyLoss(torch.FloatTensor(d))
        # loss_fct = BCEWithLogitsLoss()
        # print('logits',logits)
        loss = loss_fct(logits.view(-1, len(d)), labels.view(-1))
        # print('num_labels',logits.view(-1, self.num_labels))
        # print('labels.view(-1)',labels.view(-1).numpy())
        # print(labels.view(-1).item())
        # print('loss',loss)
        return loss,logits
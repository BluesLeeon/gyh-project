from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import pandas as pd
import torch
import attention

class BertForSequenceClassification(nn.Module):
    def __init__(self, config,num_labels):
        super(BertForSequenceClassification, self).__init__()

        # hidden_dropout_prob = 0.1
        self.dropout = nn.Dropout(0.1)
        self.softmax=nn.Softmax(dim=1)
        self.sigmond=nn.Sigmoid()
        self.fake_classifier = nn.Linear(768, num_labels) #768->34


        self.classifier_gate = nn.Linear(3*147*147,num_labels) #768->34



        # # self.classifier = nn.Linear(config.hidden_size+num_labels , num_labels) #68->34
        # self.classifierboth = nn.Linear(2*num_labels, num_labels)  # 68->34
        self.final = nn.Linear(2*num_labels, num_labels) #768->34

        self.mix = nn.Linear(768+3*147*147, num_labels)  # 768->34

        self.fake1 = nn.Linear(3*147*147, num_labels)
        # self.final = nn.Linear(2*num_labels, num_labels)

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
        pooled_output1=None,
        label_u=None
    ):
        # print(pooled_output1.shape)

        pooled_output1 = self.dropout(pooled_output1)
        fake_predict = self.fake_classifier(pooled_output1)  # 假的输出
        gate_predict = self.sigmond(self.classifier_gate(label_u))
        # fake_predict.add(gate_predict)
        final = torch.cat([fake_predict, gate_predict], 1)
        # gate_out_put = gate_predict.mul(shangyiceng_output)

        # input_classifiar_tensor = torch.cat([pooled_output1,gate_out_put],1)
        # input_classifiar_tensor = torch.add(input=pooled_output1, alpha=1, other=vec_1)

        # print(final.shape)
        logits = self.final(final)
        # print(logits.shape)
        # d=[  1., 30,184.1,166.3,45.2,271.3,245.5,271.3,139.35,234.36,17.9,
        #      166.3,47.3, 147.3, 286.4,  55.4,396.6, 139.3,  95.5, 234.4]
        # d = [1 for i in range(20)]
        # weight = torch.FloatTensor(d)
        # loss_fct = CrossEntropyLoss(torch.FloatTensor(d))

        loss_fct = CrossEntropyLoss()
        # loss_fct = BCEWithLogitsLoss()
        # print('logits',logits)
        #
        # print('logits', labels)
        loss = loss_fct(logits.view(-1,41), labels.view(-1))
        # print('num_labels',logits.view(-1, self.num_labels))
        # print('labels.view(-1)',labels.view(-1).numpy())
        # print(labels.view(-1).item())
        # print('loss',loss)
        return loss,logits
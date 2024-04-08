from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, \
    confusion_matrix
import Model
import torch
import pandas as pd
import utils
from tqdm import tqdm
import main
config=main.Config()
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW



def evaluate(y_pred, y_true):
    print("Precision: ", precision_score(y_true, y_pred, average='micro'))
    print("Recall:", recall_score(y_true, y_pred, average='micro'))
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("F1 score: ", f1_score(y_true, y_pred, average='micro'))
    # print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def validation(model,i,x, y,  batch_size, lastpre):
    model.eval()
    batch_valid = utils.batch_iter(x, y,  batch_size, lastpre)
    eval_acc = 0
    eval_loss = 0
    count = 0
    for x_batch, y_batch, lastpre in batch_valid:
        x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)
        y_batch = torch.from_numpy(y_batch).type(torch.LongTensor)
        # attn_mask = torch.from_numpy(attn_mask).type(torch.LongTensor)
        lastpre = torch.from_numpy(lastpre).type(torch.FloatTensor)


        outputs = model(
            pooled_output1=x_batch,
            token_type_ids=None,
            labels=y_batch,
            label_u=lastpre
        )

        loss = outputs[0]
        num_correct = (torch.max(outputs[1], 1)[1] == y_batch.data).sum()

        acc = (100 * num_correct) / len(x_batch)
        eval_loss += loss.item()
        eval_acc += acc.item()
        count += 1
    eval_loss = eval_loss / count
    eval_acc = eval_acc / count
    # torch.save(model.state_dict(), config.model_save_path)
    return eval_loss, eval_acc


def train(optimizer,schedule,model,i,x, y,  batch_size, last,
          text2,test_label,lastpre_test):
    test_result = []
    for epoch in tqdm(range(config.epoch)):
        model.train()
        total_epoch_loss = 0
        total_epoch_acc = 0

        batch_train = utils.batch_iter(x, y, batch_size, last)
        # batch_train = utils.batch_iter(
        #     text1,
        #     train_label,
        #     config.batch_size,
        #     lastpre_train)
        steps = 0

        for x_batch, y_batch, lastpre in batch_train:
            x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)
            y_batch = torch.from_numpy(y_batch).type(torch.LongTensor)
            lastpre = torch.from_numpy(lastpre).type(torch.FloatTensor)
            model.zero_grad()
            # print('la',lastpre)

            outputs = model(
                pooled_output1=x_batch,
                token_type_ids=None,
                labels=y_batch,
                shangyiceng_output=None,
            label_u=lastpre
            )
            loss = outputs[0]

            total_epoch_loss += loss

            num_correct = (torch.max(outputs[1], 1)[1] == y_batch.data).sum()

            acc = (100 * num_correct) / len(x_batch)
            total_epoch_acc += acc.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()

            steps += 1
        # print('Epoch: {0:02}'.format(epoch + 1))
        # import numpy as np
        # vec_1 = []
        # df = pd.read_csv(r'data/vec/luvec1.csv', header=None)
        # for ii in df.values:
        #     for j in ii:
        #         vec_1.append(j)
        # text2 = np.load("data/lutoushe/gyhuse_test_bertcls.npy")
        # all_text2=[]
        # for ii in text2:
        #     s=[]
        #     for j in ii:
        #         s.append(j)
        #     s.extend(vec_1)
        #     all_text2.append(s)

        # lastpre_test=[i[768::] for i in text2]
        # all_text2=[i[0:768] for i in text2]

        eval_loss, eval_acc = validation(model,i,text2,test_label,1,lastpre_test)

        path_for_best = config.model_save_path + str(i) + str(eval_acc)

        # if (i == 1 and eval_acc > 95)  or (i == 2 and eval_acc > 92):
        #     torch.save(model.state_dict(), path_for_best)
        #     break
        # if epoch == config.epoch - 1:
        #     torch.save(model.state_dict(), path_for_best)
        if (epoch - 1) % 100 == 0:

            print(epoch)
            print('Train Loss: {0:.3f}'.format(total_epoch_loss / steps),
                  'Train Acc: {0:.3f}%'.format(total_epoch_acc / steps))
            print(
                'Validation Loss: {0:.3f}'.format(eval_loss),
                'Validation Acc: {0:.3f}%'.format(eval_acc))
            print(max(test_result))
        test_result.append(eval_acc)
    print(test_result)
    print('max(test_result)为',max(test_result))
    return path_for_best

def predict(path_for_best, model,i,x, y,  batch_size, lastpre):
    def batch_iter(x, y, batch_size, lastpre_test):
        """
        :param x: data
        :param y: label
        :param batch_size: how many samples in one single batch
        :return: a batch of data
        """
        data_len = len(x)
        num_batch = int(
            data_len / batch_size) if data_len % batch_size == 0 else int(data_len / batch_size) + 1
        for i in range(num_batch):
            start_id = i
            end_id = (i + 1)
            yield x[start_id:end_id], y[start_id:end_id], lastpre_test[start_id:end_id]

    batch_test = batch_iter(x, y,  batch_size, lastpre)
    y_pred = []
    y_true = []
    model.load_state_dict(torch.load(path_for_best))
    model.eval()

    for x_batch, y_batch, lastpre in batch_test:
        x_batch = torch.from_numpy(x_batch).type(torch.FloatTensor)
        y_batch = torch.from_numpy(y_batch).type(torch.LongTensor)

        lastpre = torch.from_numpy(lastpre).type(torch.FloatTensor)
        # print(lastpre)
        if i == 2:
            outputs = model(
                pooled_output1=x_batch,
                token_type_ids=None,

                labels=y_batch,
                shangyiceng_output=lastpre
            )
        else:
            outputs = model(
                pooled_output1=x_batch,
                token_type_ids=None,
                labels=y_batch,
                shangyiceng_output=None
            )
        prediction = outputs[1]
        y_pred_batch = torch.max(prediction, 1)[1]
        y_pred.extend(y_pred_batch.tolist())
        y_true.extend(y_batch.tolist())

    evaluate(y_pred, y_true)
    df_raw = pd.read_csv(config.train_path)
    y_pred_label = []
    y_name = 'y{}'.format(i)

    if i == 1:
        y_pred = y_pred
    elif i == 2:
        y_pred = [i + 11 for i in y_pred]

    for num in range(len(y_pred)):
        y_pred_label.append(df_raw[df_raw[y_name] == y_pred[num]]['topic'].values.tolist()[0])
    return y_pred, y_pred_label
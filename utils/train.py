import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter, confusion_matrix_fn

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        #for metric in self.metrics:
        #    metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()

        confusion_matrix_all_categories={}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, filename in iterator:
                x, y = x.to(self.device), y.to(self.device)

                loss, y_pred = self.batch_update(x, y)

                y_pred=torch.exp(y_pred)
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {"my_loss": loss_meter.mean}
                logs.update(loss_logs)

                cm = confusion_matrix_fn(y_pred, y)

                for category in cm:
                    if category not in confusion_matrix_all_categories:
                        confusion_matrix_all_categories[category] = cm[category]
                    else:
                        for tp_tn_fp_fn in cm[category]:
                            confusion_matrix_all_categories[category][tp_tn_fp_fn] += cm[category][tp_tn_fp_fn]

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        IoUs=[]
        for category in confusion_matrix_all_categories:
            TP = confusion_matrix_all_categories[category]["TP"]
            TN = confusion_matrix_all_categories[category]["TN"]
            FP = confusion_matrix_all_categories[category]["FP"]
            FN = confusion_matrix_all_categories[category]["FN"]
            IoUs.append( (TP) / (TP + FN + FP))

        ACCs=[]
        for category in confusion_matrix_all_categories:
            TP = confusion_matrix_all_categories[category]["TP"]
            TN = confusion_matrix_all_categories[category]["TN"]
            FP = confusion_matrix_all_categories[category]["FP"]
            FN = confusion_matrix_all_categories[category]["FN"]
            ACCs.append( (TP+TN) / (TP + FN + FP+TN))

        logs["confusion_matrix"]=confusion_matrix_all_categories
        logs["category_IoU"]=IoUs

        TP=TN=FP=FN=0
        for category in confusion_matrix_all_categories:
            TP += confusion_matrix_all_categories[category]["TP"]
            TN += confusion_matrix_all_categories[category]["TN"]
            FP += confusion_matrix_all_categories[category]["FP"]
            FN += confusion_matrix_all_categories[category]["FN"]

        for metric in self.metrics:
            if metric == "mean_iou_score":
                logs["mean_iou_score"]=sum(IoUs)/len(IoUs)
            if metric == "global_iou_score":
                logs["global_iou_score"]=(TP)/(TP+FN+FP)
            if metric == "accuracy":
                logs["accuracy"]=sum(ACCs)/len(ACCs)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, baseline_method=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.baseline_method=baseline_method

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()

        if self.baseline_method == True:
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        else:
            prediction, attentions = self.model.forward(x)
            loss = self.loss(prediction, y, attentions)

        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, baseline_method=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.baseline_method=baseline_method

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            if self.baseline_method == True:
                prediction = self.model.forward(x)
                loss = self.loss(prediction, y)
            else:
                prediction, attentions = self.model.forward(x)
                loss = self.loss(prediction, y, attentions)
        return loss, prediction

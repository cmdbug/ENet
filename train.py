from tqdm import tqdm

from data.utils import get_image_from_prediction, get_image_from_label
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
from models.pointrend import point_sample

from args import USE_POINT_REND


class Train:
    """Performs the training of ``model`` given a training dataset data
    loader, the optimizer, and the loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to train.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - optim (``Optimizer``): The optimization algorithm.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, optim, criterion, metric, device, summary=None):
        self.model = model
        self.data_loader = data_loader
        self.optim = optim
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.summary = summary

    def run_epoch(self, iteration_loss=False, epoch=None):
        """Runs an epoch of training.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float).

        """
        self.model.train()
        epoch_loss = 0.0
        self.metric.reset()
        with tqdm(total=len(self.data_loader), desc=f'Train', unit='it', ncols=None) as pbar:
            for step, batch_data in enumerate(self.data_loader):
                # Get the inputs and labels
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                # Forward propagation
                if not USE_POINT_REND:
                    outputs = self.model(inputs)

                    # Loss computation
                    loss = self.criterion(outputs, labels)
                else:
                    # === [ pointrend start] ===
                    outputs = self.model(inputs)
                    pred = F.interpolate(outputs["coarse"], inputs.shape[-2:], mode="bilinear", align_corners=True)
                    seg_loss = F.cross_entropy(pred, labels, ignore_index=12)

                    gt_points = point_sample(
                        labels.float().unsqueeze(1),
                        outputs["points"],
                        mode="nearest",
                        align_corners=False
                    ).squeeze_(1).long()
                    points_loss = F.cross_entropy(outputs["rend"], gt_points, ignore_index=12)

                    loss = seg_loss + points_loss
                    # === [ pointrend end] ===


                # Backpropagation
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Keep track of loss for current epoch
                epoch_loss += loss.item()

                # Keep track of the evaluation metric
                if not USE_POINT_REND:
                    self.metric.add(outputs.detach(), labels.detach())
                else:
                    self.metric.add(outputs['x'].detach(), labels.detach())

                pbar.set_postfix(**{'Step': step, 'Iteration loss': loss.item()})
                pbar.update()
                if iteration_loss:
                    print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

            if self.summary:
                self.summary.add_image('train/1_image', inputs[0], global_step=epoch)
                self.summary.add_image('train/2_label', get_image_from_label(torch.unsqueeze(labels[0], dim=0)),
                                       global_step=epoch)
                if not USE_POINT_REND:
                    color_predictions = get_image_from_prediction(torch.unsqueeze(outputs[0], dim=0))
                else:
                    color_predictions = get_image_from_prediction(torch.unsqueeze(outputs['x'][0], dim=0))
                self.summary.add_image('train/3_predict', color_predictions, global_step=epoch)
                # === [ pointrend start] ===
                self.model.eval()
                outputs = self.model(inputs)
                if not USE_POINT_REND:
                    color_predictions = get_image_from_prediction(torch.unsqueeze(outputs[0], dim=0))
                else:
                    color_predictions = get_image_from_prediction(torch.unsqueeze(outputs['fine'][0], dim=0))
                self.summary.add_image('train/4_pointrend', color_predictions, global_step=epoch)
                self.model.train()
                # === [ pointrend end] ===


        return epoch_loss / len(self.data_loader), self.metric.value()

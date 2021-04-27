import torch
from tqdm import tqdm
from args import USE_POINT_REND


class Test:
    """Tests the ``model`` on the specified test dataset using the
    data loader, and loss criterion.

    Keyword arguments:
    - model (``nn.Module``): the model instance to test.
    - data_loader (``Dataloader``): Provides single or multi-process
    iterators over the dataset.
    - criterion (``Optimizer``): The loss criterion.
    - metric (```Metric``): An instance specifying the metric to return.
    - device (``torch.device``): An object representing the device on which
    tensors are allocated.

    """

    def __init__(self, model, data_loader, criterion, metric, device, summary=None):
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metric = metric
        self.device = device
        self.summary = summary

    def run_epoch(self, iteration_loss=False, epoch=None):
        """Runs an epoch of validation.

        Keyword arguments:
        - iteration_loss (``bool``, optional): Prints loss at every step.

        Returns:
        - The epoch loss (float), and the values of the specified metrics

        """
        self.model.eval()
        epoch_loss = 0.0
        self.metric.reset()
        with tqdm(total=len(self.data_loader), desc=f'Test', unit='it', ncols=None) as pbar:
            for step, batch_data in enumerate(self.data_loader):
                # Get the inputs and labels
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                with torch.no_grad():
                    # Forward propagation
                    if not USE_POINT_REND:
                        outputs = self.model(inputs)
                    else:
                        outputs = self.model(inputs)['fine']

                    # Loss computation
                    loss = self.criterion(outputs, labels)

                # Keep track of loss for current epoch
                epoch_loss += loss.item()

                # Keep track of evaluation the metric
                self.metric.add(outputs.detach(), labels.detach())

                pbar.set_postfix(**{'Step': step, 'Iteration loss': loss.item()})
                pbar.update()
                if iteration_loss:
                    print("[Step: %d] Iteration loss: %.4f" % (step, loss.item()))

        return epoch_loss / len(self.data_loader), self.metric.value()

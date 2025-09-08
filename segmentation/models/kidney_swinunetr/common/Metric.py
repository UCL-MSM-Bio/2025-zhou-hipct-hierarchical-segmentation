import torch
import numpy as np
import config

class Metric():
    def __init__(self, reduction = 'mean'):
        self.reduciton = reduction

    def set_input(self, y_true, y_pred):
        '''
        :param y_true: 
            torch.Tensor [b, 32, 256, 256]
        :param y_pred: 
            torch.Tensor [b, 32, 256, 256]
        '''
        self.y_true_batch = None # for the batch-format y_true
        self.y_pred_batch = None # for the batch-format y_pred
        self.y_true_case = None # for the case-format y_true
        self.y_pred_case = None # for the case-format y_pred
        self.shape = y_true.shape
        self.num_class = config.NUM_CLASSES

        if len(self.shape) == 4: # the batch-format
            assert y_true.shape == y_pred.shape, 'the shape of y_true and y_pred should be same!'
            self.y_true_batch = y_true.clone().detach().cpu().numpy()
            self.y_pred_batch = y_pred.clone().detach().cpu().numpy()

        if len(self.shape) == 3: # the no batch-fromat
            assert y_true.shape == y_pred.shape, 'the shape of y_true and y_pred should be same!'
            self.y_true_case = y_true.clone().detach().cpu().numpy()
            self.y_pred_case = y_pred.clone().detach().cpu().numpy()

    def dice_for_case(self):

        out = np.zeros((self.num_class)) # [,2]
        assert len(self.y_pred_case.shape) == 3, 'the input for dice_for_batch should has 3 dims'

        try:
            # Compute glomeruli Dice > 0 is the glomeruli
            #print('pred max: ',self.y_pred_case.max())
            #print('pred sum: ',self.y_pred_case.sum())
            #print('true max: ',self.y_true_case.max())
            #print('true sum: ',self.y_true_case.sum())

            gl_pd = np.greater(self.y_pred_case, 0.0)
            gl_gt = np.greater(self.y_true_case, 0.0)

            #print('gl_pd: ', gl_pd.sum())
            #print('gl_gt: ', gl_gt.sum())

            gl_dice = (2 * np.logical_and(gl_pd, gl_gt).sum() + 1) / (
                    (gl_pd**2).sum() + (gl_gt**2).sum() + 1
            )
        except ZeroDivisionError:
            return 0.0, 0.0
        
        out[0] = gl_dice # glomeruli dice; out[1] = 0.0
        return out

    def dice_for_batch(self):
        assert len(self.shape) == 4, 'the input for dice_for_batch should has 4 dims'
        # out = torch.rand(self.shape[0], self.num_class)
        out = np.zeros((self.shape[0], self.num_class)) # [b, c]

        for batch_index in range(self.shape[0]):
            self.y_true_case = self.y_true_batch[batch_index]
            self.y_pred_case = self.y_pred_batch[batch_index]
            out[batch_index] = self.dice_for_case()

        return out


if __name__ == '__main__':
    num_class = 3
    hight, width, = 100, 100
    batch = 4
    # define the object of Metric class
    metricer = Metric()
    # for testing the dice_for_case
    prediciton = torch.randint(low=0, high=num_class, size=(num_class, hight, width),dtype=torch.float)
    label = torch.randint(low=0, high=num_class, size=(num_class, hight, width),dtype=torch.float)
    metricer.set_input(label, prediciton)
    print('test for the dice_for_case')
    print('dice: ', metricer.dice_for_case())

    prediciton = torch.randint(low=0, high=num_class, size=(batch, num_class, hight, width),dtype=torch.float)
    label = torch.randint(low=0, high=num_class, size=(batch, num_class, hight, width),dtype=torch.float)
    metricer.set_input(label, prediciton)
    print('test for the dice_for_batch')
    print('dice: ', metricer.dice_for_batch())

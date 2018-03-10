from keras.callbacks import *

class TrapezoidalLR(Callback):
    """This callback implements a learning rate policy based on trapezoid schedule.
    For convenience, here we call it as TrapezoidalLR or TLR.
    TLR increases the learning rate as training progresses,
    then eventually decrease to make loss annealed for convergence to a minimum.
    This is similar to cyclical learning rate (CLR) schedule,
    but unlike CLR, TLR doesn't cyclically repeat this process.
    
    # Papers:
    - [1] Chen Xing, Devansh Arpit, Christos Tsirigotis, Yoshua Bengio,
      A Walk with SGD, 2018. https://arxiv.org/abs/1802.08770
    - [2] Leslie N. Smith, Cyclical Learning Rates for Training Neural Networks, 2015.
      https://arxiv.org/abs/1506.01186
    
    # Example
        ```python
            tlr = TrapezoidalLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000,  anneal_start_epoch=100 - 1)
            model.fit(X_train, Y_train, callbacks=[tlr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            tlr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            tlr = TrapezoidalLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000, anneal_start_epoch=100 - 1,
                                scale_fn=tlr_fn, scale_mode='zero2one')
            model.fit(X_train, Y_train, callbacks=[tlr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the lower boundary.
        max_lr: upper boundary.
        step_size: number of training iterations for ramp-up or down.
        anneal_start_epoch: number of epoch to start annealing at final stage of training.
        scale_fn: custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
        scale_mode: {'zero2one', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            $[0, iteration/step_size]$ or training iterations since start of ramp up or down.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006,
                 step_size=2000, anneal_start_epoch=10-1,
                 scale_fn=None, scale_mode='zero2one'):
        super(TrapezoidalLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.anneal_start_epoch = anneal_start_epoch
        if scale_fn == None:
            self.scale_fn = lambda x: x
            self.scale_mode = 'zero2one'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.tlr_iterations = 0.
        self.trn_iterations = 0.
        self.anneal_start_iteration = 0
        self.anneal_started = False
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None, new_anneal_start_epoch=None):
        """Resets iteration settings.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        if new_anneal_start_epoch != None:
            self.anneal_start_epoch = new_anneal_start_epoch
        self.tlr_iterations = 0.
        self.anneal_start_iteration = 0
        self.anneal_started = False
        
    def tlr(self):
        iteration = self.tlr_iterations - self.anneal_start_iteration if self.anneal_started else self.tlr_iterations
        iteration = min(iteration, self.step_size)        
        x = np.abs(iteration/self.step_size)
        if self.scale_mode == 'zero2one':
            x = self.scale_fn(x)
        else:
            x = self.scale_fn(iteration)
        if self.anneal_started:
            x = 1 - x
        return (1 - x) * self.base_lr + x * self.max_lr

    def on_train_begin(self, logs={}):
        logs = logs or {}

        K.set_value(self.model.optimizer.lr, self.tlr())   
        
    def on_epoch_begin(self, epoch, logs=None):
        if not self.anneal_started and self.anneal_start_epoch <= epoch:
            self.anneal_started = True
            self.anneal_start_iteration = self.tlr_iterations

    def on_batch_end(self, batch, logs=None):        
        logs = logs or {}
        self.trn_iterations += 1
        self.tlr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.tlr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
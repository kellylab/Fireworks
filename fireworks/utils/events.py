from ignite.handlers import ModelCheckpoint, Timer
import logging
import visdom

logging.getLogger(name=__name__)

"""
This code is largely adapted from the demos from the PyTorch and Ignite documentation.
See https://github.com/pytorch/ignite for more information.
"""

def visdom_loss_handler(modules_dict, model_name):
    """
    Attaches plots and metrics to trainer.
    This handler creates or connects to an environment on a running Visdom dashboard and creates a line plot that tracks the loss function of a
    training loop as a function of the number of iterations. This can be attached to an Ignite Engine, and the training closure must
    have 'loss' as one of the keys in its return dict for this plot to be made.
    See documentation for Ignite (https://github.com/pytorch/ignite) and Visdom (https://github.com/facebookresearch/visdom) for more information.
    """

    tim = Timer()
    tim.attach( trainer,
                start=Events.STARTED,
                step=Events.ITERATION_COMPLETED,
    )

    vis = visdom.Visdom(env=environment)
    def create_plot_window(vis, xlabel, ylabel, title):
        return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

    train_loss_window = create_plot_window(vis, '#Iterations', 'Loss', description)
    log_interval = 10

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration -1)
        if iter % log_interval == 0:
            logging.info("Epoch[{}] Iteration: {} Time: {} Loss: {:.2f}".format(
                engine.state.epoch, iter, str(datetime.timedelta(seconds=int(tim.value()))), engine.state.output
            ))
        vis.line(X=np.array([engine.state.iteration]),
                 Y=np.array([engine.state.output]),
                 update='append',
                 win=train_loss_window)

    save_interval = 50
    handler = ModelCheckpoint('/tmp/models', model_name, save_interval = save_interval, n_saved=5, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, handler, modules_dict)

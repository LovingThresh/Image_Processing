import tensorflow.keras as keras


# checkpoint
# 目前存在的问题是只能输出最后一个checkpoints
# 2021/9/20 问题已经解决，目前可以输出指定的checkpoint了
# 测试一下git的合并功能
class CheckpointSaver(keras.callbacks.Callback):
    def __init__(self, manager):
        super(CheckpointSaver, self).__init__()
        self.manager = manager

    def on_epoch_end(self, epoch, logs=None):
        self.manager.save()


lr_schedule_E = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.96
)

# 2、PiecewiseConstantDecay(分段常数衰减)
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
lr_schedule_P = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=boundaries,
    values=values
)

# 3、PolynomialDecay(多项式衰减)
starter_learning_rate = 0.1
end_learning_rate = 0.01
decay_steps = 10000
lr_schedule_Pi = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=starter_learning_rate,
    decay_steps=decay_steps,
    end_learning_rate=end_learning_rate,
    power=0.5
)

# 4、InverseTimeDecay(逆时间衰减)
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 0.5
lr_schedule_I = keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate
)

# 监视验证集损失函数动态调整

DynamicLearningRate = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=10, verbose=0, mode='auto',
    min_delta=0.0001, cooldown=0, min_lr=0
)

EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')

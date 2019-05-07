from Utils import visualize_model, create_model, train, TensorBoardImage
from Data_generator import data_generator
import os

model_dir = os.path.join('..','models')
layers = {'Layer_0':{'Encoding':[16,32],'Decoding':[32,16,8]},
          'Base':{'Encoding':[64]}}
model_desc = 'Shallow_net5' # Name of your model
# The numbers inside are the number of filter banks, you can have mulitple filter banks per layer

visualize_model(layers, model_desc, vol_size=(256,256,1))
train_dir = os.path.join('..','data','train','aug')
train_generator = data_generator(train_dir)
print('We have ' + str(len(train_generator)) + ' images available')

learning_rate = 0.001 # Rate at which our gradients will change during each back propogation, typically in range of 1e-2 to 1e-5
number_of_epochs = 100 # The number of epochs to be trained, one epoch means that you have seen the entirety of your dataset
                      # However, since we defined steps per epoch this might not apply
steps_per_epoch = len(train_generator)
loss_function = 'categorical_crossentropy'
batch_normalization = True

model, callbacks = create_model(layers, model_desc, batch_norm=batch_normalization, data_generator=train_generator)
new_call_back = TensorBoardImage("",os.path.join('..','Tensorboard_models',model_desc),train_generator)
tensorboard_output = os.path.join('..', 'Tensorboard_models', model_desc)
if not os.path.exists(tensorboard_output):
    os.makedirs(tensorboard_output)
# new_call_back = new_tensorboard(log_dir=tensorboard_output, batch_size=2, write_graph=True, write_grads=False,
#                           write_images=True, update_freq='epoch', histogram_freq=0)
# new_call_back.set_training_model(train_generator)

# callbacks = callbacks + [new_call_back]
train(model, train_generator, callbacks, learning_rate, number_of_epochs, steps_per_epoch,loss_function)

xxx = 1
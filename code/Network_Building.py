import sys
from keras.models import Model
from keras.layers import Conv2D, Activation, Input, UpSampling2D, Concatenate, BatchNormalization, MaxPooling2D
from keras.layers import LeakyReLU
from keras.initializers import RandomNormal


class UNet_Core():
    def __init__(self, activation='elu', alpha=0.2, input_size=(256,256,1), visualize=False,batch_normalization=True):
        self.filters = (3,3)
        self.activation = activation
        self.alpha = alpha
        self.pool_size = (2,2)
        self.input_size = input_size
        self.visualize=visualize
        self.batch_normalization = batch_normalization

    def conv_block(self,output_size,x, strides=1):
        x = Conv2D(output_size, self.filters, activation=None, padding='same',
                   name=self.desc, strides=strides)(x)
        if self.activation != 'LeakyReLU':
            x = LeakyReLU(self.alpha)(x)
        else:
            x = Activation(self.activation)(x)
        self.layer += 1
        if not self.visualize and self.batch_normalization:
            x = BatchNormalization()(x)
        return x

    def get_unet(self, layers_dict):
        self.layers_names = []
        layers = 0
        for name in layers_dict:
            if name.find('Base') != 0:
                layers += 1
        for i in range(layers):
            self.layers_names.append('Layer_' + str(i))
        self.layers_names.append('Base')
        x = input_image = Input(shape=self.input_size, name='Input')
        self.layer = 0
        layer_vals = {}
        self.desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = layers_dict[layer]['Encoding']
            for i in range(len(all_filters)):
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i) if strides == 1 else layer + '_Strided_Conv' + str(i)
                if strides == 2 and layer_index not in layer_vals:
                    layer_vals[layer_index] = x
                x = self.conv_block(all_filters[i], x, strides=strides)
                layer_vals[layer_index] = x
            layer_index += 1
            x = MaxPooling2D(name=layer + '_Max_Pooling')(x)
        if 'Base' in layers_dict:
            strides = 1
            all_filters = layers_dict['Base']['Encoding']
            for i in range(len(all_filters)):
                self.desc = 'Base_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, strides=strides)
        self.desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            layer_index -= 1
            all_filters = layers_dict[layer]['Decoding']
            x = UpSampling2D(name='Upsampling' + str(self.layer) + '_UNet')(x)
            x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[layer_index]])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x)
        x = Conv2D(2, self.filters, activation='softmax', padding='same',
                   name='output')(x)
        model = Model(inputs=input_image, outputs=x)
        self.created_model = model


class new_model(object):
    def __init__(self, layers, image_size=(256,256, 1), indexing='ij',batch_normalization=False,visualize=False):
        self.indexing = indexing
        UNet_Core_class = UNet_Core(input_size=image_size, batch_normalization=batch_normalization, visualize=visualize)
        UNet_Core_class.get_unet(layers)
        self.model = UNet_Core_class.created_model

def main():
    pass

if __name__ == '__main__':
    main()

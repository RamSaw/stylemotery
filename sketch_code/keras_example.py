import keras
from keras.models import Model
import keras.backend as K
from keras.engine.topology import Layer, InputLayer, Input, Node
from keras.layers import merge, Dense, TimeDistributed, LSTM
import theano.tensor as T
import theano

import numpy as np


class RecursiveInput(InputLayer):
    def __init__(self, input_shape=None, batch_input_shape=None,
                 input_dtype=None, input_tensor=None, name=None):

        # Create the shared variable which will represent the input
        input_t = K.ones(batch_input_shape)
        self.bis = batch_input_shape
        super(RecursiveInput, self).__init__(input_shape, batch_input_shape,
                                             input_dtype, input_t, name)

    def getTensor(self):
        t = self.inbound_nodes[0].output_tensors
        if len(t) == 1:
            return t[0]
        else:
            return t

    def getTensorValues(self):
        return K.eval(self.getTensor())

    def resetTensorValues(self):
        K.set_value(self.getTensor(), np.ones(self.bis))


class RecursiveOutput(Layer):
    def __init__(self, recursiveInput, **kwargs):
        super(RecursiveOutput, self).__init__(**kwargs)
        if type(recursiveInput) != RecursiveInput:
            raise Exception('Expected input layer to be of the'
                            'type RecursiveInput.')
        self.recursiveInput = recursiveInput
        self.stateful = True
        self.targetTensor = recursiveInput.getTensor()

    def build(self, input_shape):
        self.built = True

    # Identity, updates the input tensor to the output of
    # the network
    def call(self, inputs, mask=None):
        self.updates = [(self.targetTensor, inputs)]
        return inputs


## Just needed to override the Container
## Constructor at the point where
## Computable layers are calculated
class RecursiveModel(Model):
    def __init__(self, input, output, recursiveInput, name=None):
        # handle name argument
        self.recursiveInput = recursiveInput
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        # Container-specific properties
        if type(input) in {list, tuple}:
            self.inputs = list(input)  # tensor or list of tensors
        else:
            self.inputs = [input]
        if type(output) in {list, tuple}:
            self.outputs = list(output)
        else:
            self.outputs = [output]

        # check for redundancy in inputs:
        inputs_set = set(self.inputs)
        if len(inputs_set) != len(self.inputs):
            raise Exception('The list of inputs passed to the model '
                            'is redundant. All inputs should only appear once.'
                            ' Found: ' + str(self.inputs))

        # list of initial layers (1 to 1 mapping with self.inputs,
        # hence the same layer might appear twice)
        self.input_layers = []
        # TODO: probably useless because input layers must be Input layers (node_indices = [0], tensor_indices = [0])
        self.input_layers_node_indices = []
        self.input_layers_tensor_indices = []
        # list of layers (1 to 1 mapping with self.inputs,
        # hence the same layer might appear twice)
        self.output_layers = []
        # TODO: probably useless
        self.output_layers_node_indices = []
        self.output_layers_tensor_indices = []
        # all layers in order of horizontal graph traversal.
        # Entries are unique. Includes input and output layers.
        self.layers = []

        # this is for performance optimization
        # when calling the Container on new inputs.
        # every time the Container is called on a set on input tensors,
        # we compute the output tensors,
        # output masks and output shapes in one pass,
        # then cache them here. When of of these output is queried later,
        # we retrieve it from there instead of recomputing it.
        self._output_mask_cache = {}
        self._output_tensor_cache = {}
        self._output_shape_cache = {}

        # arguments validation
        for x in self.inputs:
            # check that x is a Keras tensor
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise Exception('Input tensors to a ' + cls_name + ' ' +
                                'must be Keras tensors. Found: ' + str(x) +
                                ' (missing Keras metadata).')
            # check that x is an input tensor
            layer, node_index, tensor_index = x._keras_history
            if len(layer.inbound_nodes) > 1 or (layer.inbound_nodes and layer.inbound_nodes[0].inbound_layers):
                cls_name = self.__class__.__name__
                warnings.warn(cls_name + ' inputs must come from '
                                         'a Keras Input layer, '
                                         'they cannot be the output of '
                                         'a previous non-Input layer. '
                                         'Here, a tensor specified as '
                                         'input to "' + self.name +
                              '" was not an Input tensor, '
                              'it was generated by layer ' +
                              layer.name + '.\n'
                                           'Note that input tensors are '
                                           'instantiated via `tensor = Input(shape)`.\n'
                                           'The tensor that caused the issue was: ' +
                              str(x.name))
        for x in self.outputs:
            if not hasattr(x, '_keras_history'):
                cls_name = self.__class__.__name__
                raise Exception('Output tensors to a ' + cls_name + ' must be '
                                                                    'Keras tensors. Found: ' + str(x))
        # build self.output_layers:
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            self.output_layers.append(layer)
            self.output_layers_node_indices.append(node_index)
            self.output_layers_tensor_indices.append(tensor_index)

        # fill in the output mask cache
        masks = []
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer.inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        mask_cache_key = ','.join([str(id(x)) for x in self.inputs])
        mask_cache_key += '_' + ','.join([str(id(x)) for x in masks])
        masks = []
        for x in self.outputs:
            layer, node_index, tensor_index = x._keras_history
            node = layer.inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        if len(masks) == 1:
            mask = masks[0]
        else:
            mask = masks
        self._output_mask_cache[mask_cache_key] = mask

        # build self.input_layers:
        for x in self.inputs:
            layer, node_index, tensor_index = x._keras_history
            # it's supposed to be an input layer, so only one node
            # and one tensor output
            assert node_index == 0
            assert tensor_index == 0
            self.input_layers.append(layer)
            self.input_layers_node_indices.append(node_index)
            self.input_layers_tensor_indices.append(tensor_index)

        # build self.input_names and self.output_names
        self.input_names = []
        self.output_names = []
        for layer in self.input_layers:
            self.input_names.append(layer.name)
        for layer in self.output_layers:
            self.output_names.append(layer.name)

        self.internal_input_shapes = [x._keras_shape for x in self.inputs]
        self.internal_output_shapes = [x._keras_shape for x in self.outputs]

        # container_nodes: set of nodes included in the graph
        # (not all nodes included in the layers are relevant to the current graph).
        container_nodes = set()  # ids of all nodes relevant to the Container
        nodes_depths = {}  # map {node: depth value}
        layers_depths = {}  # map {layer: depth value}

        def make_node_marker(node, depth):
            return str(id(node)) + '-' + str(depth)

        def build_map_of_graph(tensor, seen_nodes=set(), depth=0,
                               layer=None, node_index=None, tensor_index=None):
            '''This recursively updates the maps nodes_depths,
            layers_depths and the set container_nodes.
            Does not try to detect cycles in graph (TODO?)
            # Arguments
                tensor: some tensor in a graph
                seen_nodes: set of node ids ("{layer.name}_ib-{node_index}")
                    of nodes seen so far. Useful to prevent infinite loops.
                depth: current depth in the graph (0 = last output).
                layer: layer from which `tensor` comes from. If not provided,
                    will be obtained from `tensor._keras_history`.
                node_index: node index from which `tensor` comes from.
                tensor_index: tensor_index from which `tensor` comes from.
            '''
            if not layer or node_index is None or tensor_index is None:
                layer, node_index, tensor_index = tensor._keras_history
            node = layer.inbound_nodes[node_index]

            # prevent cycles
            seen_nodes.add(make_node_marker(node, depth))

            node_key = layer.name + '_ib-' + str(node_index)
            # update container_nodes
            container_nodes.add(node_key)
            # update nodes_depths
            node_depth = nodes_depths.get(node)
            if node_depth is None:
                nodes_depths[node] = depth
            else:
                nodes_depths[node] = max(depth, node_depth)
            # update layers_depths
            previously_seen_depth = layers_depths.get(layer)
            if previously_seen_depth is None:
                current_depth = depth
            else:
                current_depth = max(depth, previously_seen_depth)
            layers_depths[layer] = current_depth

            # propagate to all previous tensors connected to this node
            for i in range(len(node.inbound_layers)):
                x = node.input_tensors[i]
                layer = node.inbound_layers[i]
                node_index = node.node_indices[i]
                tensor_index = node.tensor_indices[i]
                next_node = layer.inbound_nodes[node_index]
                # use node_marker to prevent cycles
                node_marker = make_node_marker(next_node, current_depth + 1)
                if node_marker not in seen_nodes:
                    build_map_of_graph(x, seen_nodes, current_depth + 1,
                                       layer, node_index, tensor_index)

        for x in self.outputs:
            seen_nodes = set()
            build_map_of_graph(x, seen_nodes, depth=0)

        # build a map {depth: list of nodes with this depth}
        nodes_by_depth = {}
        for node, depth in list(nodes_depths.items()):
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)

        # build a map {depth: list of layers with this depth}
        layers_by_depth = {}
        for layer, depth in list(layers_depths.items()):
            if depth not in layers_by_depth:
                layers_by_depth[depth] = []
            layers_by_depth[depth].append(layer)

        # get sorted list of layer depths
        depth_keys = list(layers_by_depth.keys())
        depth_keys.sort(reverse=True)

        # set self.layers and self.layers_by_depth
        layers = []
        for depth in depth_keys:
            layers_for_depth = layers_by_depth[depth]
            # container.layers needs to have a deterministic order
            layers_for_depth.sort(key=lambda x: x.name)
            for layer in layers_for_depth:
                layers.append(layer)
        self.layers = layers
        self.layers_by_depth = layers_by_depth

        # get sorted list of node depths
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        # check that all tensors required are computable.
        # computable_tensors: all tensors in the graph
        # that can be computed from the inputs provided
        computable_tensors = [self.recursiveInput]
        for x in self.inputs:
            computable_tensors.append(x)

        layers_with_complete_input = []  # to provide a better error msg
        for depth in depth_keys:
            for node in nodes_by_depth[depth]:
                layer = node.outbound_layer
                if layer:
                    for x in node.input_tensors:
                        if x not in computable_tensors:
                            raise Exception(
                                'Graph disconnected: '
                                'cannot obtain value for tensor ' +
                                str(x) + ' at layer "' + layer.name + '". '
                                                                      'The following previous layers '
                                                                      'were accessed without issue: ' +
                                str(layers_with_complete_input))
                    for x in node.output_tensors:
                        computable_tensors.append(x)
                    layers_with_complete_input.append(layer.name)

        # set self.nodes and self.nodes_by_depth
        self.container_nodes = container_nodes
        self.nodes_by_depth = nodes_by_depth

        # ensure name unicity, which will be crucial for serialization
        # (since serialized nodes refer to layers by their name).
        all_names = [layer.name for layer in self.layers]
        for name in all_names:
            if all_names.count(name) != 1:
                raise Exception('The name "' + name + '" is used ' +
                                str(all_names.count(name)) +
                                ' times in the model. ' +
                                'All layer names should be unique.')

        # layer parameters
        # the new container starts with a single inbound node
        # for its inputs, and no outbound nodes.
        self.outbound_nodes = []  # will be appended to by future calls to __call__
        self.inbound_nodes = []  # will be appended to below, and by future calls to __call__
        # create the node linking internal inputs to internal outputs
        Node(outbound_layer=self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=self.inputs,
             output_tensors=self.outputs,
             # no container-level masking for now
             input_masks=[None for _ in self.inputs],
             output_masks=[None for _ in self.outputs],
             input_shapes=[x._keras_shape for x in self.inputs],
             output_shapes=[x._keras_shape for x in self.outputs])
        self.built = True
        self.supports_masking = False
        # the following are implemented as property functions:
        # self.constraints
        # self.regularizers
        # self.trainable_weights
        # self.non_trainable_weights
        # self.input_spec


if __name__ == "__main__":

    # Network One
    # A normal input
    batch_input_shape = (1, 4, 5,)
    input = Input(batch_shape=batch_input_shape, name="In")

    # The mutable input
    rinput = RecursiveInput(batch_input_shape=batch_input_shape, name="RecursiveIn")
    rTensor = rinput.getTensor()

    m = merge([input, rTensor], mode='concat', name="Merge")

    d = LSTM(5, return_sequences=True)(m)
    rout = RecursiveOutput(rinput, name="RecursiveOut")(d)

    routskip = RecursiveOutput(rinput, name="RecursiveOut")(m)

    model1 = RecursiveModel(input=input, output=rout, recursiveInput=rTensor)
    model12 = RecursiveModel(input=input, output=routskip, recursiveInput=rTensor)

    i = np.random.rand(1, 4, 5)
    l = 0
    for k in i[0]:
        print(l, ": ", k)
        l = l + 1

    print()
    " Inital TensorVals", model1.get_layer(name="RecursiveIn").getTensorValues()
    model1.predict(i)
    print()
    "Final TensorVals", model1.get_layer(name="RecursiveIn").getTensorValues()

    print()
    "\n With merge, skipping LSTM"
    model12.get_layer(name="RecursiveIn").resetTensorValues()
    print()
    model12.predict(i)

    # Network Two
    # Network runs the imput one timestep at a time to
    # achieve the desired dependecies in the input data
    bis = (1, 1, 5,)
    i2 = Input(batch_shape=bis, name="In")

    rinput2 = RecursiveInput(batch_input_shape=bis, name="RecursiveIn")
    rTensor2 = rinput2.getTensor()

    # Inputs are TensorVariable and TensorSharedVariable, output is a TensorVariable
    m2 = merge([i2, rTensor2], mode='concat', name="Merge")

    d2 = LSTM(5, return_sequences=True)(m2)
    rout2 = RecursiveOutput(rinput2, name="RecursiveOut")(d2)

    ## Weights are the same to compare with model1
    model2 = RecursiveModel(input=i2, output=rout2, recursiveInput=rTensor2)
    model2.set_weights(model1.get_weights())

    print("\nModel2 test of chained timesteps")
    print("This is the intended behaviour.")
    split_inputs = []
    for k in i[0]:
        split_inputs.append(np.asarray([[k]]))

    l = 0
    for j in split_inputs:
        print(l, ": ", j)
        l = l + 1
        model2.predict(j)
        print("\t TVState: ", model2.get_layer(name="RecursiveIn").getTensorValues())

        # print "\n Model2 test of timesteps applied individually"
        # l =0
        # for j in split_inputs:
        #    model2.get_layer(name="RecursiveIn").resetTensorValues()
        #    print l, ": ", j
        #    l = l+l
        #    model2.predict(j)
        #    print model2.get_layer(name="RecursiveIn").getTensorValues()
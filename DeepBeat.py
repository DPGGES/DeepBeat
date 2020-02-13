# Copyright 2020 The DeepBeat Author.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train and evaluate a drums LSTM model. Project based on Magenta"""
import os
import magenta
import tensorflow as tf

from magenta.models.shared import sequence_generator
from magenta.models.shared import events_rnn_model
import magenta.music as mm
from magenta.pipelines import drum_pipelines
from magenta.music.protobuf import generator_pb2
from magenta.music.protobuf import music_pb2

from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import metrics as contrib_metrics
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import slim as contrib_slim
from tensorflow.python.util import nest as tf_nest

import numbers
import six
import numpy as np
from tensorflow.contrib import training as contrib_training
import functools
import ast
import time

"""
###################################################################
#                       Configuration                             #                                                                                                    
###################################################################
"""
# default configuraction. Based on Magenta
default_configs = {'drum_kit':
    events_rnn_model.EventSequenceRnnConfig(
        magenta.music.protobuf.generator_pb2.GeneratorDetails(
            id='drum_kit',
            description='Drums RNN with multiple drums and binary counters.'
        ),
        magenta.music.LookbackEventSequenceEncoderDecoder(
            magenta.music.MultiDrumOneHotEncoding(),
            lookback_distances=[],
            binary_counter_bits=6),
        contrib_training.HParams(
            batch_size=64,
            rnn_layer_sizes=[256, 256],
            dropout_keep_prob=0.5,
            attn_length=32,
            clip_norm=3,
            learning_rate=0.001))
}

# TensorFlow Flags.
#Flags to configure evaluation and training. Spanish edition
Flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run_dir', '/tmp/Deepbeat/logdir/run1',
                           'Directorio  donde se encuentran los checkpoinst y '
                           'los análisis de los eventos.Se crearán en directorios separados para '
                           'el entrenamiento y la evaluación. '
                           'Se pueden guardar diferentes entrenamientos/evaluaciones '
                           'en el mismo directorio. Si se desea usar TensorBoard'
                           'aputar al directorio run_dir.' )
tf.app.flags.DEFINE_string('SE', '',
                           'Directorio donde se encuentran los archivos TFRecord '
                           'para  el entrenamiento o la evaluación. ')
tf.app.flags.DEFINE_integer('num_training_steps', 0,
                            'Específica el número de steps en el entrenamiento. '
                            'Dejar valor 0 para terminar de forma manual.')
tf.app.flags.DEFINE_string('modo_trabajo', 'train',
                           'train: entrentamiento '
                           'eval: evaluación '
                           'generate: inferencia ')
tf.app.flags.DEFINE_integer('num_eval_batches', 64,
                            'Especifica el número de batches en la evaluación. ')
tf.app.flags.DEFINE_string( 'hparams', '','Especifica parametros de red. '
                            'Separados con coma, listados con "name=valor, name2=valor".')

#Flags for inference. Spanish edition.
tf.app.flags.DEFINE_string('output_dir', '/tmp/drums_rnn/generated',
                           'Directorio donde los archivos MIDI se guardarán.')
tf.app.flags.DEFINE_integer('num_outputs', 1,
                            'Número de pistas de audio que se generarán (en formato MIDI)')
tf.app.flags.DEFINE_integer('num_steps', 128,
                            'El total de número de steps que se generan por pista(track). '
                            'El número 128 implica que son 8 compases')
tf.app.flags.DEFINE_string( 'primer_drums', '',
                            'Es una representación en texto sobre el patron rítmico el cual el modelo'
                            'se tiene que basar. Por ejemplo: "[(36,42),(),(),(),(42,),(),(),()]" ')

tf.app.flags.DEFINE_float('qpm', None,
                          'Este valor es los "quarters per minute", indica el tempo para reproducir la salida. '
                          'Si el valor es None, el valor por defecto será 120.')

tf.app.flags.DEFINE_float('temperature', 1.0,
                          'Representa la  aleatoriedad de las pistas de drums generadas. 1.0 utiliza el inalterado'
                          'Valores mayores que 1 generar pistas más aleatorias y valores menores que 1 menos '
                          'aleatorias.')

#@TODO:
tf.app.flags.DEFINE_string('primer_midi', '',
                            'Directorio del archivo MIDI que sustituye al flag prime_midi.'
                            'Si no se especifica, la pista sera creada desde el principio.')

"""
###################################################################
#                       Classes                                   #                                                                                                    
###################################################################
"""
"""
#######################################################################################
# Class: EvalLoggingTensorHook
# Description: Class declared for evaluation
#                                                                            
#######################################################################################
"""


class EvalLoggingTensorHook(tf.estimator.LoggingTensorHook):


    def begin(self):
        # Reset timer.
        self._timer.update_last_triggered_step(0)
        super(EvalLoggingTensorHook, self).begin()

    def before_run(self, run_context):
        self._iter_count += 1
        return super(EvalLoggingTensorHook, self).before_run(run_context)

    def after_run(self, run_context, run_values):
        super(EvalLoggingTensorHook, self).after_run(run_context, run_values)
        self._iter_count -= 1


"""
#######################################################################################
# Clase: DrumsRnnSequenceGenerator
# Arguments: 
#            model: Instance of DrumsRnnModel.
#            details: A generator_pb2.GeneratorDetails for this generator.
#            steps_per_quarter: What precision to use when quantizing the melody. How
#                               many steps per quarter note.
#            checkpoint: Where to search for the most recent model checkpoint. Mutually
#                        exclusive with `bundle`.
#            bundle: A GeneratorBundle object that includes both the model checkpoint
#                    and metagraph. Mutually exclusive with `checkpoint`.
# Return:
#           generated_sequence: Secuencia generada
# Description: 
#              Shared Melody RNN generation code as a SequenceGenerator interface. Adaptation.                                                                
#######################################################################################
"""
#TODO @DGG : use the model generator below

class DrumsRnnSequenceGenerator(sequence_generator.BaseSequenceGenerator):


    def __init__(self, model, details, steps_per_quarter=4, checkpoint=None,
                 bundle=None):

        #TODO @DGG : NO USAR FUNCIONES DE LA CLASE PADRE.
        super(DrumsRnnSequenceGenerator, self).__init__(
            model, details, checkpoint, bundle)
        self.steps_per_quarter = steps_per_quarter

    def _generate(self, input_sequence, generator_options):
        if len(generator_options.input_sections) > 1:
            raise sequence_generator.SequenceGeneratorError(
                'This model supports at most one input_sections message, but got %s' %
                len(generator_options.input_sections))
        if len(generator_options.generate_sections) != 1:
            raise sequence_generator.SequenceGeneratorError(
                'This model supports only 1 generate_sections message, but got %s' %
                len(generator_options.generate_sections))

        if input_sequence and input_sequence.tempos:
            qpm = input_sequence.tempos[0].qpm
        else:
            qpm = mm.DEFAULT_QUARTERS_PER_MINUTE
        steps_per_second = mm.steps_per_quarter_to_steps_per_second(
            self.steps_per_quarter, qpm)

        generate_section = generator_options.generate_sections[0]
        if generator_options.input_sections:
            input_section = generator_options.input_sections[0]
            primer_sequence = mm.trim_note_sequence(
                input_sequence, input_section.start_time, input_section.end_time)
            input_start_step = mm.quantize_to_step(
                input_section.start_time, steps_per_second, quantize_cutoff=0.0)
        else:
            primer_sequence = input_sequence
            input_start_step = 0

        if primer_sequence.notes:
            last_end_time = max(n.end_time for n in primer_sequence.notes)
        else:
            last_end_time = 0
        if last_end_time > generate_section.start_time:
            raise sequence_generator.SequenceGeneratorError(
                'Got GenerateSection request for section that is before the end of '
                'the NoteSequence. This model can only extend sequences. Requested '
                'start time: %s, Final note end time: %s' %
                (generate_section.start_time, last_end_time))

        # Quantize the priming sequence.
        quantized_sequence = mm.quantize_note_sequence(
            primer_sequence, self.steps_per_quarter)
        # Setting gap_bars to infinite ensures that the entire input will be used.
        extracted_drum_tracks, _ = drum_pipelines.extract_drum_tracks(
            quantized_sequence, search_start_step=input_start_step, min_bars=0,
            gap_bars=float('inf'), ignore_is_drum=True)
        assert len(extracted_drum_tracks) <= 1

        start_step = mm.quantize_to_step(
            generate_section.start_time, steps_per_second, quantize_cutoff=0.0)
        # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
        # always rounds down. This avoids generating a sequence that ends at 5.0
        # seconds when the requested end time is 4.99.
        end_step = mm.quantize_to_step(
            generate_section.end_time, steps_per_second, quantize_cutoff=1.0)

        if extracted_drum_tracks and extracted_drum_tracks[0]:
            drums = extracted_drum_tracks[0]
        else:
            # If no drum track could be extracted, create an empty drum track that
            # starts 1 step before the request start_step. This will result in 1 step
            # of silence when the drum track is extended below.
            steps_per_bar = int(
                mm.steps_per_bar_in_quantized_sequence(quantized_sequence))
            drums = mm.DrumTrack([],
                                 start_step=max(0, start_step - 1),
                                 steps_per_bar=steps_per_bar,
                                 steps_per_quarter=self.steps_per_quarter)

        # Ensure that the drum track extends up to the step we want to start
        # generating.
        drums.set_length(start_step - drums.start_step)

        # Extract generation arguments from generator options.
        arg_types = {
            'temperature': lambda arg: arg.float_value,
            'beam_size': lambda arg: arg.int_value,
            'branch_factor': lambda arg: arg.int_value,
            'steps_per_iteration': lambda arg: arg.int_value
        }
        args = dict((name, value_fn(generator_options.args[name]))
                    for name, value_fn in arg_types.items()
                    if name in generator_options.args)

        generated_drums = self._model.generate_drum_track(
            end_step - drums.start_step, drums, **args)
        generated_sequence = generated_drums.to_sequence(qpm=qpm)
        assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
        return generated_sequence


"""
#######################################################################################
# Class: DrumsRnnModel
# Arguments:
#              num_steps: The integer length in steps of the final drum track, after
#                         generation. Includes the primer.
#              primer_drums: The primer drum track, a DrumTrack object.
#              temperature: A float specifying how much to divide the logits by
#                           before computing the softmax. Greater than 1.0 makes drum tracks more
#                           random, less than 1.0 makes drum tracks less random.
#              beam_size: An integer, beam size to use when generating drum tracks via
#                         beam search.
#              branch_factor: An integer, beam search branch factor to use.
#              steps_per_iteration: An integer, number of steps to take per beam search
#                                   iteration.
#        
# Returns:
#              The generated DrumTrack object (which begins with the provided primer drum
#                  track).
# Description:  Class declared for inference
#                                                                         
#######################################################################################
"""


class DrumsRnnModel(events_rnn_model.EventSequenceRnnModel):


    def generate_drum_track(self, num_steps, primer_drums, temperature=1.0,
                            beam_size=1, branch_factor=1, steps_per_iteration=1):

        return self._generate_events(num_steps, primer_drums, temperature,
                                     beam_size, branch_factor, steps_per_iteration)


"""
###################################################################
#                Auxiliar  function                              #                                                                                                    
###################################################################
#######################################################################################
# Function: make_rnn_cell
#  Arguments:
#           rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
#                            RNN.
#           dropout_keep_prob: The float probability to keep the output of any given
#                              sub-cell.
#           attn_length: The size of the attention vector.
#           base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
#           residual_connections: Whether or not to use residual connections (via
#                                 tf.contrib.rnn.ResidualWrapper).
#
#  Returns:
#      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
#
# Description:  make a RNN Cell. Adaptation
#                                                                                   
#######################################################################################
"""


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=contrib_rnn.BasicLSTMCell,
                  residual_connections=False):
    cells = []
    for i in range(len(rnn_layer_sizes)):
        cell = base_cell(rnn_layer_sizes[i])
        if attn_length and not cells:
            # Add attention wrapper to first layer.
            cell = contrib_rnn.AttentionCellWrapper(
                cell, attn_length, state_is_tuple=True)
        if residual_connections:
            cell = contrib_rnn.ResidualWrapper(cell)
            if i == 0 or rnn_layer_sizes[i] != rnn_layer_sizes[i - 1]:
                cell = contrib_rnn.InputProjectionWrapper(cell, rnn_layer_sizes[i])
        cell = contrib_rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)

    cell = contrib_rnn.MultiRNNCell(cells)

    return cell


"""
#######################################################################################
# Function: get_TF_graph
# Arguments:
#            mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
#                the graph.
#            config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
#                to use.
#            sequence_example_file_paths: A list of paths to TFRecord files containing
#                tf.train.SequenceExample protos. Only needed for training and
#                evaluation.
#
#  Returns:
#           A function that builds the TF ops when called.
#
#  Raises:
#           ValueError: If mode is not 'train', 'eval', or 'generate'.
#                   :param mode:
#                   :param config:
#                   :param sequence_example_file_paths:
#                   :return:
#  Description:
            Returns a function that builds the TensorFlow graph.                                                                          
#######################################################################################
"""



def get_TF_graph(mode, config, sequence_example_file_paths=None):


    hparams = config.hparams
    if hparams.use_cudnn:
        raise ValueError('cuDNN no esta actualmente soportado.')
    tf.compat.v1.logging.info('hparams = %s', hparams.values())

    encoder_decoder = config.encoder_decoder
    input_size = encoder_decoder.input_size
    num_classes = encoder_decoder.num_classes
    no_event_label = encoder_decoder.default_event_label

    def CrearTFGraph():
        """Builds the Tensorflow graph."""
        inputs, labels, lengths = None, None, None
        if mode in ('train', 'eval'):
            if isinstance(no_event_label, numbers.Number):
                label_shape = []
            else:
                label_shape = [len(no_event_label)]
            inputs, labels, lengths = magenta.common.get_padded_batch(
                sequence_example_file_paths, hparams.batch_size, input_size,
                label_shape=label_shape, shuffle=mode == 'train')

        elif mode == 'generate':
            inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                                 input_size])

        dropout_keep_prob = 1.0 if mode == 'generate' else hparams.dropout_keep_prob

        if hparams.use_cudnn:
            print("Cudnn no disponible")

        else:
            cell = make_rnn_cell(
                hparams.rnn_layer_sizes,
                dropout_keep_prob=dropout_keep_prob,
                attn_length=hparams.attn_length,
                residual_connections=hparams.residual_connections)

            initial_state = cell.zero_state(hparams.batch_size, tf.float32)

            outputs, final_state = tf.nn.dynamic_rnn(
                cell, inputs, sequence_length=lengths, initial_state=initial_state,
                swap_memory=True)

        outputs_flat = magenta.common.flatten_maybe_padded_sequences(
            outputs, lengths)
        if isinstance(num_classes, numbers.Number):
            num_logits = num_classes
        else:
            num_logits = sum(num_classes)
        logits_flat = contrib_layers.linear(outputs_flat, num_logits)

        if mode in ('train', 'eval'):

            labels_flat = magenta.common.flatten_maybe_padded_sequences(
                labels, lengths)

            if isinstance(num_classes, numbers.Number):
                softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels_flat, logits=logits_flat)
                predictions_flat = tf.argmax(logits_flat, axis=1)
            else:
                logits_offsets = np.cumsum([0] + num_classes)
                softmax_cross_entropy = []
                predictions = []
                for i in range(len(num_classes)):
                    softmax_cross_entropy.append(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=labels_flat[:, i],
                            logits=logits_flat[
                                   :, logits_offsets[i]:logits_offsets[i + 1]]))
                    predictions.append(
                        tf.argmax(logits_flat[
                                  :, logits_offsets[i]:logits_offsets[i + 1]], axis=1))
                predictions_flat = tf.stack(predictions, 1)

            correct_predictions = tf.cast(
                tf.equal(labels_flat, predictions_flat), tf.float32)
            event_positions = tf.cast(tf.not_equal(labels_flat, no_event_label), tf.float32)
            no_event_positions = tf.cast(tf.equal(labels_flat, no_event_label), tf.float32)

            # Compute the total number of time steps across all sequences in the
            # batch. For some models this will be different from the number of RNN
            # steps.
            def batch_labels_to_num_steps(batch_labels, lengths):
                num_steps = 0
                for labels, length in zip(batch_labels, lengths):
                    num_steps += encoder_decoder.labels_to_num_steps(labels[:length])
                return np.float32(num_steps)

            num_steps = tf.py_func(
                batch_labels_to_num_steps, [labels, lengths], tf.float32)

            if mode == 'train':

                loss = tf.reduce_mean(softmax_cross_entropy)
                perplexity = tf.exp(loss)
                accuracy = tf.reduce_mean(correct_predictions)
                event_accuracy = (
                        tf.reduce_sum(correct_predictions * event_positions) /
                        tf.reduce_sum(event_positions))
                no_event_accuracy = (
                        tf.reduce_sum(correct_predictions * no_event_positions) /
                        tf.reduce_sum(no_event_positions))

                loss_per_step = tf.reduce_sum(softmax_cross_entropy) / num_steps
                perplexity_per_step = tf.exp(loss_per_step)

                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hparams.learning_rate)

                train_op = contrib_slim.learning.create_train_op(
                    loss, optimizer, clip_gradient_norm=hparams.clip_norm)
                tf.compat.v1.add_to_collection('train_op', train_op)

                vars_to_summarize = {
                    'loss': loss,
                    'metrics/perplexity': perplexity,
                    'metrics/accuracy': accuracy,
                    'metrics/event_accuracy': event_accuracy,
                    'metrics/no_event_accuracy': no_event_accuracy,
                    'metrics/loss_per_step': loss_per_step,
                    'metrics/perplexity_per_step': perplexity_per_step,
                }
            elif mode == 'eval':
                vars_to_summarize, update_ops = contrib_metrics.aggregate_metric_map({
                    'loss':
                        tf.metrics.mean(softmax_cross_entropy),
                    'metrics/accuracy':
                        tf.metrics.accuracy(labels_flat, predictions_flat),
                    'metrics/per_class_accuracy':
                        tf.metrics.mean_per_class_accuracy(labels_flat,
                                                           predictions_flat,
                                                           num_classes),
                    'metrics/event_accuracy':
                        tf.metrics.recall(event_positions, correct_predictions),
                    'metrics/no_event_accuracy':
                        tf.metrics.recall(no_event_positions, correct_predictions),
                    'metrics/loss_per_step':
                        tf.metrics.mean(
                            tf.reduce_sum(softmax_cross_entropy) / num_steps,
                            weights=num_steps),
                })
                for updates_op in update_ops.values():
                    tf.add_to_collection('eval_ops', updates_op)

                # Perplexity is just exp(loss) and doesn't need its own update op.
                vars_to_summarize['metrics/perplexity'] = tf.exp(
                    vars_to_summarize['loss'])
                vars_to_summarize['metrics/perplexity_per_step'] = tf.exp(
                    vars_to_summarize['metrics/loss_per_step'])

            for var_name, var_value in six.iteritems(vars_to_summarize):
                tf.compat.v1.summary.scalar(var_name, var_value)
                tf.add_to_collection(var_name, var_value)

        elif mode == 'generate':
            temperature = tf.placeholder(tf.float32, [])
            if isinstance(num_classes, numbers.Number):
                softmax_flat = tf.nn.softmax(
                    tf.div(logits_flat, tf.fill([num_classes], temperature)))
                softmax = tf.reshape(
                    softmax_flat, [hparams.batch_size, -1, num_classes])
            else:
                logits_offsets = np.cumsum([0] + num_classes)
                softmax = []
                for i in range(len(num_classes)):
                    sm = tf.nn.softmax(
                        tf.div(
                            logits_flat[:, logits_offsets[i]:logits_offsets[i + 1]],
                            tf.fill([num_classes[i]], temperature)))
                    sm = tf.reshape(sm, [hparams.batch_size, -1, num_classes[i]])
                    softmax.append(sm)

            tf.add_to_collection('inputs', inputs)
            tf.add_to_collection('temperature', temperature)
            tf.add_to_collection('softmax', softmax)
            # Flatten state tuples for metagraph compatibility.
            for state in tf_nest.flatten(initial_state):
                tf.add_to_collection('initial_state', state)
            for state in tf_nest.flatten(final_state):
                tf.add_to_collection('final_state', state)

    return CrearTFGraph


"""
#######################################################################################
# Function: Entrenamiento
# Args:
#    construye_grafo: Una función que construye un grafo de operaciones.
#    train_dir: El directorio donde los checkpoints del entrenamiento serán cargados
#    num_training_steps: El número de steps en el  entrenamiento .
# Description: Runs the evaluation loop.
#                                                                            
#######################################################################################
"""


def Entrenamiento(construye_grafo, train_dir, num_training_steps=None):
    summary_frequency = 10
    save_checkpoint_secs = 60
    checkpoints_to_keep = 10
    keep_checkpoint_every_n_hours = 1
    master = ''
    task = 0
    num_ps_tasks = 0

    with tf.Graph().as_default():
        with tf.device(tf.compat.v1.train.replica_device_setter(num_ps_tasks)):
            tf.compat.v1.logging.set_verbosity('ERROR')
            construye_grafo()

            global_step = tf.compat.v1.train.get_or_create_global_step()
            loss = tf.compat.v1.get_collection('loss')[0]
            perplexity = tf.compat.v1.get_collection('metrics/perplexity')[0]
            accuracy = tf.compat.v1.get_collection('metrics/accuracy')[0]
            train_op = tf.compat.v1.get_collection('train_op')[0]

            logging_dict = {
                'Global Step': global_step,
                'Loss': loss,
                'Perplexity': perplexity,
                'Accuracy': accuracy
            }

            hooks = [
                tf.estimator.NanTensorHook(loss),
                tf.estimator.LoggingTensorHook(
                    logging_dict, every_n_iter=summary_frequency),
                tf.estimator.StepCounterHook(
                    output_dir=train_dir, every_n_steps=summary_frequency)
            ]

            if num_training_steps:
                hooks.append(tf.estimator.StopAtStepHook(num_training_steps))

            scaffold = tf.compat.v1.train.Scaffold(
                saver=tf.compat.v1.train.Saver(
                    max_to_keep=checkpoints_to_keep,
                    keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours))
            tf.compat.v1.logging.set_verbosity('INFO')
            tf.compat.v1.logging.info('Comenzando ciclo de entrenamiento...')
            contrib_training.train(
                train_op=train_op,
                logdir=train_dir,
                scaffold=scaffold,
                hooks=hooks,
                save_checkpoint_secs=save_checkpoint_secs,
                save_summaries_steps=summary_frequency,
                master=master,
                is_chief=task == 0)
            tf.compat.v1.logging.info('Entrenamiento completado.')


"""
#######################################################################################
# Function: Evaluacion
# Args:
#    build_graph_fn: Una funcion que construye un grafo de operaciones.
#    train_dir: El directorio donde los checkpoints del entrenamiento serán cargados
#    eval_dir: El directorio donde el resumen de la evaluación sera guardado
#    num_batches: El número de batches que se usan para cada step en la evaluacion
#    timeout_secs: El número de segundo que debe esperar para analizar otro checkpoint.
#  Raises:
#    ValueError: si `num_batches` es menor o igual que 0.
#
# Description: Runs the evaluation loop.
#                                                                            
#######################################################################################
"""


def Evaluacion(build_graph_fn, train_dir, eval_dir, num_batches,
               timeout_secs=300):

    tf.compat.v1.logging.set_verbosity('INFO')
    if num_batches <= 0:
        raise ValueError(
            '`num_batches` must be greater than 0. Check that the batch size is '
            'no larger than the number of records in the eval set.')
    with tf.Graph().as_default():
        # Creamos un modelo igual que el del entrenamiento
        build_graph_fn()
        # Define the summaries to write:
        global_step = tf.train.get_or_create_global_step()
        loss = tf.get_collection('loss')[0]
        perplexity = tf.get_collection('metrics/perplexity')[0]
        accuracy = tf.get_collection('metrics/accuracy')[0]
        eval_ops = tf.get_collection('eval_ops')

        logging_dict = {
            'Global Step': global_step,
            'Loss': loss,
            'Perplexity': perplexity,
            'Accuracy': accuracy
        }
        hooks = [
            EvalLoggingTensorHook(logging_dict, every_n_iter=num_batches),
            contrib_training.StopAfterNEvalsHook(num_batches),
            contrib_training.SummaryAtEndHook(eval_dir),
        ]
        # names_to_values = contrib_training.evaluate_once(
        #     checkpoint_path=train_dir,
        #     eval_ops=eval_ops,
        #     final_ops=logging_dict,
        #     hooks=hooks,
        #     config=None)
        # for name in names_to_values:
        #     print('Metric %s has value %f.' % (name, names_to_values[name]))
        contrib_training.evaluate_repeatedly(
            train_dir,
            eval_ops=eval_ops,
            hooks=hooks,
            eval_interval_secs=2,
            timeout=timeout_secs)


"""
#######################################################################################
# Function: get_checkpoint
#
# Description: Get the training dir or checkpoint path to be used by the model.
#                                                                            
#######################################################################################
"""

def get_checkpoint():


    if Flags.run_dir:
        train_dir = os.path.join(os.path.expanduser(Flags.run_dir), 'train')
        return train_dir
    elif Flags.checkpoint_file:
        return os.path.expanduser(Flags.checkpoint_file)
    else:
        return None


"""
#######################################################################################
# Function: get_generator_map
#   Arguments:
#               Binds the `config` argument so that the arguments match the
#               BaseSequenceGenerator class constructor.: 
# Returns: 
#                Map from the generator ID to its SequenceGenerator class creator with a
#                bound `config` argument.
# Description: Returns a map from the generator ID to a SequenceGenerator class creator
#                                                                            
#######################################################################################
"""

def get_generator_map():

    def create_sequence_generator(config, **kwargs):
        return DrumsRnnSequenceGenerator(
            default_configs.DrumsRnnModel(config), config.details,
            steps_per_quarter=config.steps_per_quarter, **kwargs)

    return {key: functools.partial(create_sequence_generator, config)
            for (key, config) in default_configs.items()}

"""
#######################################################################################
# Function: run_with_flags
# Arguments: 
#           generator: The DrumsRnnSequenceGenerator to use for generation.
#                        Check TensorFlow Flags for mor infor
# 
# Description: Creates a drum track and save it as MIDI file
#                                                                            
#######################################################################################
"""


def run_with_flags(generator):

    if not Flags.output_dir:
        tf.logging.fatal('--output_dir required')
        return
    Flags.output_dir = os.path.expanduser(Flags.output_dir)

    primer_midi = None
    if Flags.primer_midi:
        primer_midi = os.path.expanduser(Flags.primer_midi)

    if not tf.gfile.Exists(Flags.output_dir):
        tf.gfile.MakeDirs(Flags.output_dir)

    primer_sequence = None
    qpm = Flags.qpm if Flags.qpm else magenta.music.DEFAULT_QUARTERS_PER_MINUTE
    if Flags.primer_drums:
        primer_drums = magenta.music.DrumTrack(
            [frozenset(pitches)
             for pitches in ast.literal_eval(Flags.primer_drums)])
        primer_sequence = primer_drums.to_sequence(qpm=qpm)
    elif primer_midi:
        primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
        if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
            qpm = primer_sequence.tempos[0].qpm
    else:
        tf.logging.warning(
            'No priming sequence specified. Defaulting to a single bass drum hit.')
        primer_drums = magenta.music.DrumTrack([frozenset([36])])
        primer_sequence = primer_drums.to_sequence(qpm=qpm)

    # Derive the total number of seconds to generate based on the QPM of the
    # priming sequence and the num_steps flag.
    seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
    total_seconds = Flags.num_steps * seconds_per_step

    # Specify start/stop time for generation based on starting generation at the
    # end of the priming sequence and continuing until the sequence is num_steps
    # long.
    generator_options = generator_pb2.GeneratorOptions()
    if primer_sequence:
        input_sequence = primer_sequence
        # Set the start time to begin on the next step after the last note ends.
        if primer_sequence.notes:
            last_end_time = max(n.end_time for n in primer_sequence.notes)
        else:
            last_end_time = 0
        generate_section = generator_options.generate_sections.add(
            start_time=last_end_time + seconds_per_step,
            end_time=total_seconds)

        if generate_section.start_time >= generate_section.end_time:
            tf.logging.fatal(
                'Priming sequence is longer than the total number of steps '
                'requested: Priming sequence length: %s, Generation length '
                'requested: %s',
                generate_section.start_time, total_seconds)
            return
    else:
        input_sequence = music_pb2.NoteSequence()
        input_sequence.tempos.add().qpm = qpm
        generate_section = generator_options.generate_sections.add(
            start_time=0,
            end_time=total_seconds)
    generator_options.args['temperature'].float_value = Flags.temperature
    tf.logging.debug('input_sequence: %s', input_sequence)
    tf.logging.debug('generator_options: %s', generator_options)

    # Make the generate request num_outputs times and save the output as midi
    # files.
    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(Flags.num_outputs))
    for i in range(Flags.num_outputs):
        generated_sequence = generator.generate(input_sequence, generator_options)
        midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
        midi_path = os.path.join(Flags.output_dir, midi_filename)
        magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)

    tf.logging.info('Wrote %d MIDI files to %s',
                    Flags.num_outputs, Flags.output_dir)


"""
###################################################################
#                       Main                                      #                                                                                                    
###################################################################
"""
"""
#######################################################################################
# Function Deep Beat
# Input: ver tensorflow Flags
# Output: Modelo entrenado, Modelo evaluado, Modelo inferido
# Description: DeepBeat es una adaptación y simplificación usando el contenido de  Magenta.
#             Permite entrenar, evaluar e inferir modelos LSTM con archivos MIDI                                                               
#######################################################################################
"""


def DeepBeat(unused_argv):
    # Interal use 'DEBUG, INFO, WARN, ERROR, or FATAL.'
    tf.compat.v1.logging.set_verbosity('INFO')
    tf.compat.v1.logging.info("###### Init DeepBeat ######")

    # 1-EVALUACION DE FLAGS
    # A-Flag SequenceExample
    if (Flags.modo_trabajo not in ('generate')):
        if not Flags.SE or not tf.io.gfile.exists(Flags.SE):
            tf.compat.v1.logging.fatal('Comprobar flag: --SE ')
            return
    dir_seq_examp = tf.io.gfile.glob(
        os.path.expanduser(Flags.SE))
    tf.compat.v1.logging.info("Directorio SequenceExamples: {}".format(dir_seq_examp))

    # B-Flag Directorio de trabajo. Si no existe se crea
    dir_trabajo = os.path.expanduser(Flags.run_dir)
    train_dir = os.path.join(dir_trabajo, 'train')
    if not os.path.exists(train_dir):
        tf.io.gfile.makedirs(train_dir)
    tf.compat.v1.logging.info('Directorio de trabajo: %s', train_dir)

    # C-Flag modo de trabajo. Si no existe error
    if Flags.modo_trabajo not in ('train', 'eval', 'generate'):
        tf.compat.v1.logging.fatal('Comprobar flag: --SE , '
                                   'se desconoce {}'.format(Flags.modo_trabajo))
        return
    modo_de_trabajo = Flags.modo_trabajo

    # D-Parámetros de configuración red neuronal
    # la variable configuracion es un  objeto tipo events_rnn_model.EventSequenceRnnConfig(...)
    configuracion = default_configs['drum_kit']
    configuracion.hparams.parse(Flags.hparams)
    tf.compat.v1.logging.info("Objeto de configuración {}".format(configuracion))

    # E-Flag Construcción del grafo con una red LSTM basada en la configuracion.
    ##construye_grafo es un puntero a una funcion que crea el grafo. Se llama en la función ntrenamiento
    construye_grafo = get_TF_graph(modo_de_trabajo, configuracion, dir_seq_examp)

    ####################
    # Entrenamiento
    ####################
    if Flags.modo_trabajo in ('train'):
        # Entreno modelo
        Entrenamiento(construye_grafo, train_dir, Flags.num_training_steps)
        return
    else:
        ####################
        # Evaluación
        ####################
        if (Flags.modo_trabajo in ('eval')):
            # Se comprueba que no esté vacío el directorio con el modelo entrenado
            if not os.listdir(train_dir):
                tf.compat.v1.logging.fatal('No se encuentra modelo en :%s', train_dir)
                return
            # se comrpueba que existe un directorio para la evaluación, en caso contrario se crea
            eval_dir = os.path.join(dir_trabajo, 'eval')
            if not os.path.exists(eval_dir):
                tf.gfile.MakeDirs(eval_dir)
            tf.logging.info('Eval dir: %s', eval_dir)
            # Número de batches para la evaluación
            num_batches = (Flags.num_eval_batches)
            if num_batches == 0:
                num_batches = default_configs.hparams.batch_size
            Evaluacion(construye_grafo, train_dir, eval_dir, num_batches)
        ####################
        # Inferencia
        ####################
        else:
            config = configuracion  # igual

            # Funciona pero esta creando el modelo usando el event rnn graph
            generator = DrumsRnnSequenceGenerator(
                model=DrumsRnnModel(config),
                details=config.details,
                steps_per_quarter=config.steps_per_quarter,
                checkpoint=get_checkpoint(),
                bundle=None)

            run_with_flags(generator)
        return


def punto_de_entrada_consola():
    tf.compat.v1.app.run(DeepBeat)


if __name__ == '__main__':
    punto_de_entrada_consola()

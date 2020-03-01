
# DeepBeat

Hip-Hop is one of the music genres that has spread remarkably on an international scale in the last decade. Due to the large number of listeners worldwide, Hip-Hop is in great demand, inspiring the creation of new productions in short periods of time. This type of consumer music needs new tools that help composers and/or producers to create compositions as quickly as possible guaranteeing a high quality of the product.

**DeepBeat** is part of a master’s degree final project that aims to develop a tool and methodology to help music composers to produce drum rhythms based on Hip-Hop. DeepBeat works as a sketcher. The input will work as a first aproximation (sketch), then it will return a series of beats continuing the secuence. 

The software is based on the DrumRnn model from [Magenta](https://magenta.tensorflow.org/) and the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove). It applies language modeling to  create a drum track using Recurrent Neural Networks LSTM. The code is written in one file to help those who are starting to program in TensorFlow (version 1.5).

The drums set is composed by 9-pieces :
* Bass drum
* 3 x  Toms
* Snare drum
* Open hit-hat
* Close hit-hat
* Ride cymbals

An example created by DeepBeat can be checked in :

[Soundcloud_Music_DeepBeat](https://soundcloud.com/deep-beat-191447108/sets/deepbeat-ai)

## 1.Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 1.1 Prerequisites

First, set up your  [Magenta environment](https://github.com/tensorflow/magenta). 

###1.2 Installing the full Development Environment.

Clone this repository:
```bash
git clone https://github.com/DPGGES/DeepBeat
```

##1.3 Pre-trained models

If you want to get started right away, we have been pre-trained twelve models based on Hip-Hop:

* [pre_trained_models](https://drive.google.com/file/d/1adv6xhrYOg8sdqc4q4C9eC1h-Qu4yjht/view?usp=sharing)


**Note**, we recommend the run256b64_15_85_ model. 
It is a LSTM with two layers of 256 units. It has been trained using a batch of 64 units and a dataset with 15% for evaluation and 85% for training.

###1.4 Starting right away

```
MODEL_PATH=<absolute path of the pre_trained model>

python DeepBeat.py \
--run_dir={MODEL_PATH} \
--hparams="batch_size=64,rnn_layer_sizes=[256,256]" \
--output_dir=/tmp/deepbeat/generated \
--num_outputs=5 \
--num_steps=128 \
--primer_drums="[(36,),(),(36,42),(),(38,),(),(),(),(),(36,42),(36,42), (),(38,),(),(),(),(36,),(),(36,42),(),(38,),(),(),(),(),(36,42),(36,42),(),(38,),(),(),()]" \
--modo_trabajo=generate \
```

This will create a drum track starting with the `primer_drumms` values. The values in the list should be tuples of integer MIDI pitches, representing the drums that are played simultaneously at each step. Instead you can use a MIDI file input using `primer_midi`flag.

Starting with the division of a bar into 16 parts. The above example shows that the flag `primer_drumms` introduces two beats (beats 1 and beat 2), therefore, there are 32 parts. Each part will be played by one or more instruments nested in tuples. For example (36-"bass drum" ,42 "closet hit-hat") are played at the same time; then there is a () which means that nothing is played  and  an instrument is written with a comma (38,).


A track has a 16 division per beat (1/16). The value 'num_steps'= 128 means that DeepBeat will generate 128/16= 8 beats.

##2 Make your own Model
 

###2.1. Create your Dataset

###2.1.1. Create NoteSequences

The first step will be to convert a collection of MIDI files into NoteSequences. NoteSequences are [protocol buffers](https://developers.google.com/protocol-buffers/). In this example, we assume the NoteSequences were output to ```/tmp/notesequences.tfrecord```. Please follow the instrucctions  [here](https://github.com/tensorflow/magenta/blob/master/magenta/scripts/README.md).

The following example is the unencrypted content within the Tfrecord file created.

```
id: "/id/midi/hiphop/520425d05b6a26f7c0aaf48683a33a159bd21baf"
filename: "107_hiphop_70_fill_4-4.mid"
collection_name: "hiphop"
ticks_per_quarter: 480
time_signatures {
  numerator: 4
  denominator: 4
}
key_signatures {
}
tempos {
  qpm: 69.99998833333528
}
notes {
  pitch: 38
  velocity: 127
  start_time: 1.735714575
  end_time: 1.8232145895833334
  is_drum: true
}
```

###2.1.2. Create SequenceExamples

SequenceExamples are fed into the model during training and evaluation. Each SequenceExample will contain a sequence of inputs and a sequence of labels that represent a drum track. Run the command below to extract drum tracks from our NoteSequences and save them as SequenceExamples. Two collections of SequenceExamples will be generated, one for training, and one for evaluation, where the fraction of SequenceExamples in the evaluation set is determined by `--eval_ratio`. With an eval ratio of 0.15, 15% of the extracted drum tracks will be saved in the eval collection, and 85% will be saved in the training collection.

```
drums_rnn_create_dataset \
--config='drum_kit' \
--input=/tmp/notesequences.tfrecord \
--output_dir=/tmp/drums_rnn/sequence_examples \
--eval_ratio=0.15
```

### 2.2 Train and evaluate your model

If you have skipped the previous section you can use our data:

* [Data](https://drive.google.com/file/d/1isrnMFRxnpnVn9ZCAELxsKG-fMNRezY0/view?usp=sharing)

####2.2.1. Training

DeepBeat is very effective and is based on DrumRnn model from Magenta. Since it allows you to create a model, compile it, evaluate it and create the Beat, a series of flags need to be filled:

*	`--run_dir`: is the directory where checkpoints and TensorBoard data (for this run) will be stored.
*	`--SE`:is the TFRecord file of SequenceExamples that will be fed to the model
*	`--modo_trabajo`: train, eval or generate.
* `--num_eval_batches`: (optional) The number of evaluation examples that your model should process for each evaluation step.
*	`--num_training_steps`: (optional) is how many update steps to take before exiting the training loop. If left unspecified, the training loop will run until terminated manually.
*	`--hparams`: (optional) can be used to specify hyperparameters other than the defaults.The default values are 2 layers of 256 units each. You can also indicate among other things the size of the batch (batch_size), which default value is 64.


```
python DeepBeat.py \
--run_dir=/tmp/run1 \
--SE=/tmp/sequence_examples/training_drum_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[256,256]" \
--num_training_steps=1000 \
--modo_trabajo=train

```

####2.2.2 Evaluation
Optionally you can run an evaluation job in parallel. The flags `--run_dir`, `--hparams`, and `--num_training_steps` should all be the same values used for the training job. `--SE` should point to the separate set of eval drum tracks. Include `--modo_trabajo=eval` to make this an evaluation job, resulting in the model only being evaluated without any of the weights being updated.


```
python DeepBeat.py \
--run_dir=/tmp/run1 \
--SE=/tmp/sequence_examples/eval_drum_tracks.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[256,256]" \
--modo_trabajo=eval \
```

####2.2.3 TensorBoard
Run TensorBoard to check the training and evaluation data. It can be called after or during the training and evaluation.

```
tensorboard --logdir=/tmp/run1
```

Then go to [http://localhost:6006](http://localhost:6006) to view the TensorBoard dashboard.

###3. Generate Drum Tracks with your model

DeepBeat can generate drum tracks during or after training(recommended). Run the command below to generate a set of drum tracks using the latest checkpoint file of your trained model. The flags `--run_dir` and `--hparams` should all be the same values used for the training job.

* `--output_dir` is where the generated MIDI files will be saved. 
* `--num_outputs` is the number of drum tracks that will be generated. 
* `--num_steps` : is how long each melody will be in 16th steps (128 steps = 8 bars).
* `--primer_drums`: is the rhythmic pattern used to create the drum track. For more information, please check section 1.4.
 
```
python DeepBeat.py \
--run_dir=/tmp/run1 \
--output_dir=/tmp/run1/output1 \
--hparams="batch_size=64,rnn_layer_sizes=[256,256]" \
--modo_trabajo=generate \
--num_steps=128 \
--num_outputs=5 \
--primer_drums= “[(36, 42), (), (42,)]”
```

If you want to check out some examples created by DeepBeat, please dowload the following  file:

* [MIDI](https://drive.google.com/file/d/1BLI8MBPQXsWrncwTBXUJ3XEoXKGkwu_4/view?usp=sharing)

## 4. DeepBeat on test

To evaluate our best model objectively, an online [survey](https://drive.google.com/file/d/1I65jAlSIMIMUPsH5xbbW3iP7RA8o8hBg/view?usp=sharing) was created to be able to extract as many samples as possible. The aim of this survey is to differentiate whether the rhythmic pattern has been developed by a human or by DeepBeat. This survey was answered by various people around the globe with and without musical knowledge.The [Crowdsignal](https://crowdsignal.com/) platform was used for this purpose. 

As we have commented throughout the project, DeepBeat is a tool that allows you to sketch out rhythmic patterns from an entry. The result is a series of MIDI files, the MIDI files do not contain audio. Thus, the composition method was using the DAW [MPC live](https://www.akaipro.com/mpc-live), the generated MIDI file was imported. This sequence remained unalterable because the aim is to analyse whether it has been created by an A.I. or a person.

It was composed of different audio tracks. This exercise was repeated 5 times, so you have 5 sequences of unaltered rhythms, different from each other and with different melodies.

[Soundcloud_Music_DeepBeat](https://soundcloud.com/deep-beat-191447108/sets/deepbeat-ai)


## 5. DeepBeat on Google Colab


Please use the notebooks(check Notebooks folder) in [Google Colaboratory](https://colab.research.google.com/) to easily train and evaluate your models.





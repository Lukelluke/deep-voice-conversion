# Voice Conversion with Non-Parallel Data
## Subtitle: Speaking like Kate Winslet
> Authors: Dabi Ahn(andabi412@gmail.com), [Kyubyong Park](https://github.com/Kyubyong)(kbpark.linguist@gmail.com)

修改自：https://github.com/andabi/deep-voice-conversion/issues/64

## Samples
https://soundcloud.com/andabi/sets/voice-style-transfer-to-kate-winslet-with-deep-neural-networks

## Intro
What if you could imitate a famous celebrity's voice or sing like a famous singer?
This project started with a goal to convert someone's voice to a specific target voice.
So called, it's voice style transfer.
We worked on this project that aims to convert someone's voice to a famous English actress [Kate Winslet](https://en.wikipedia.org/wiki/Kate_Winslet)'s 
[voice](https://soundcloud.com/andabi/sets/voice-style-transfer-to-kate-winslet-with-deep-neural-networks).
We implemented a deep neural networks to achieve that and more than 2 hours of audio book sentences read by Kate Winslet are used as a dataset.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/deep-voice-conversion/master/materials/title.png" width="50%"></p>

## Model Architecture
This is a many-to-one voice conversion system.
The main significance of this work is that we could generate a target speaker's utterances without parallel data like <source's wav, target's wav>, <wav, text> or <wav, phone>, but only waveforms of the target speaker.
(To make these parallel datasets needs a lot of effort.)
All we need in this project is a number of waveforms of the target speaker's utterances and only a small set of <wav, phone> pairs from a number of anonymous speakers.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/deep-voice-conversion/master/materials/architecture.png" width="85%"></p>

The model architecture consists of two modules:
1. Net1(phoneme classification) classify someone's utterances to one of phoneme classes at every timestep.
    * Phonemes are speaker-independent while waveforms are speaker-dependent.
2. Net2(speech synthesis) synthesize speeches of the target speaker from the phones.

We applied CBHG(1-D convolution bank + highway network + bidirectional GRU) modules that are mentioned in [Tacotron](https://arxiv.org/abs/1703.10135).
CBHG is known to be good for capturing features from sequential data.

### Net1 is a classifier.
* Process: wav -> spectrogram -> mfccs -> phoneme dist.
* Net1 classifies spectrogram to phonemes that consists of 60 English phonemes at every timestep.
  * For each timestep, the input is log magnitude spectrogram and the target is phoneme dist.
* Objective function is cross entropy loss.
* [TIMIT dataset](https://catalog.ldc.upenn.edu/LDC93S1) used.
  * contains 630 speakers' utterances and corresponding phones that speaks similar sentences.
* Over 70% test accuracy

### Net2 is a synthesizer.
Net2 contains Net1 as a sub-network.
* Process: net1(wav -> spectrogram -> mfccs -> phoneme dist.) -> spectrogram -> wav
* Net2 synthesizes the target speaker's speeches.
  * The input/target is a set of target speaker's utterances.
* Since Net1 is already trained in previous step, the remaining part only should be trained in this step.
* Loss is reconstruction error between input and target. (L2 distance)
* Datasets
    * Target1(anonymous female): [Arctic](http://www.festvox.org/cmu_arctic/) dataset (public)
    * Target2(Kate Winslet): over 2 hours of audio book sentences read by her (private)
* Griffin-Lim reconstruction when reverting wav from spectrogram.

## Implementations
### Requirements
* python 2.7
* tensorflow >= 1.1
* numpy >= 1.11.1
* librosa == 0.5.1

### Settings
* sample rate: 16,000Hz
* window length: 25ms
* hop length: 5ms

### Procedure
* Train phase: Net1 and Net2 should be trained sequentially.
  * Train1(training Net1)
    * Run `train1.py` to train and `eval1.py` to test.
  * Train2(training Net2)
    * Run `train2.py` to train and `eval2.py` to test.
      * Train2 should be trained after Train1 is done!
* Convert phase: feed forward to Net2
    * Run `convert.py` to get result samples.
    * Check Tensorboard's audio tab to listen the samples.
    * Take a look at phoneme dist. visualization on Tensorboard's image tab.
      * x-axis represents phoneme classes and y-axis represents timesteps
      * the first class of x-axis means silence.

<p align="center"><img src="https://raw.githubusercontent.com/andabi/deep-voice-conversion/master/materials/phoneme_dist.png" width="30%"></p>

## Tips (Lessons We've learned from this project)
* Window length and hop length have to be small enough to be able to fit in only a phoneme.
* Obviously, sample rate, window length and hop length should be same in both Net1 and Net2.
* Before ISTFT(spectrogram to waveforms), emphasizing on the predicted spectrogram by applying power of 1.0~2.0 is helpful for removing noisy sound.
* It seems that to apply temperature to softmax in Net1 is not so meaningful.
* IMHO, the accuracy of Net1(phoneme classification) does not need to be so perfect.
  * Net2 can reach to near optimal when Net1 accuracy is correct to some extent.

## References
* ["Phonetic posteriorgrams for many-to-one voice conversion without parallel data training"](https://www.researchgate.net/publication/307434911_Phonetic_posteriorgrams_for_many-to-one_voice_conversion_without_parallel_data_training), 2016 IEEE International Conference on Multimedia and Expo (ICME)
* ["TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS"](https://arxiv.org/abs/1703.10135), Submitted to Interspeech 2017



```
name: deepvoice
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - backports=1.0=pyhd3eb1b0_2
  - backports.weakref=1.0rc1=py36_0
  - blas=1.0=mkl
  - bleach=1.5.0=py36_0
  - ca-certificates=2020.10.14=0
  - certifi=2020.12.5=py36h06a4308_0
  - cudatoolkit=8.0=3
  - cudnn=6.0.21=cuda8.0_0
  - html5lib=0.9999999=py36_0
  - importlib-metadata=2.0.0=py_1
  - intel-openmp=2020.2=254
  - ld_impl_linux-64=2.33.1=h53a641e_7
  - libedit=3.1.20191231=h14c3975_1
  - libffi=3.3=he6710b0_2
  - libgcc=7.2.0=h69d50b8_2
  - libgcc-ng=9.1.0=hdf63c60_0
  - libprotobuf=3.13.0.1=hd408876_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - markdown=3.3.3=py36h06a4308_0
  - mkl=2020.2=256
  - mkl-service=2.3.0=py36he8ac12f_0
  - mkl_fft=1.2.0=py36h23d657b_0
  - mkl_random=1.1.1=py36h0573a6f_0
  - ncurses=6.2=he6710b0_1
  - numpy=1.19.2=py36h54aff64_0
  - numpy-base=1.19.2=py36hfa32c7d_0
  - openssl=1.1.1h=h7b6447c_0
  - pip=20.3.1=py36h06a4308_0
  - protobuf=3.13.0.1=py36he6710b0_1
  - python=3.6.12=hcff3b4d_2
  - readline=8.0=h7b6447c_0
  - setuptools=51.0.0=py36h06a4308_2
  - six=1.15.0=py36h06a4308_0
  - sqlite=3.33.0=h62c20be_0
  - tensorflow-gpu=1.3.0=0
  - tensorflow-gpu-base=1.3.0=py36cuda8.0cudnn6.0_1
  - tensorflow-tensorboard=1.5.1=py36hf484d3e_1
  - tk=8.6.10=hbc83047_0
  - werkzeug=1.0.1=py_0
  - wheel=0.36.1=pyhd3eb1b0_0
  - xz=5.2.5=h7b6447c_0
  - zipp=3.4.0=pyhd3eb1b0_0
  - zlib=1.2.11=h7b6447c_3
  - pip:
    - audioread==2.1.9
    - biwrap==0.1.6
    - cffi==1.14.4
    - cycler==0.10.0
    - decorator==4.4.2
    - joblib==0.17.0
    - kiwisolver==1.3.1
    - librosa==0.6.2
    - llvmlite==0.31.0
    - matplotlib==3.3.3
    - msgpack==1.0.0
    - msgpack-numpy==0.4.7.1
    - numba==0.48.0
    - pillow==8.0.1
    - pycparser==2.20
    - pydub==0.24.1
    - pyparsing==2.4.7
    - python-dateutil==2.8.1
    - pyyaml==5.3.1
    - pyzmq==20.0.0
    - resampy==0.2.2
    - scikit-learn==0.23.2
    - scipy==1.5.4
    - soundfile==0.10.3.post1
    - tabulate==0.8.7
    - tensorboard==1.7.0
    - tensorflow-plot==0.3.0
    - tensorpack==0.9.0
    - termcolor==1.1.0
    - threadpoolctl==2.1.0
    - tqdm==4.54.1
prefix: /home/hsj/miniconda3/envs/deepvoice


***

python train1.py timit -gpu 0

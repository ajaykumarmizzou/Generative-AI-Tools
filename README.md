# Generative AI Tools

This document provides a comprehensive list of tools commonly used in generative AI. Each tool has its own strengths and is designed to perform specific tasks in the generative AI workflow.

## Machine Learning Frameworks

- [TensorFlow](https://www.tensorflow.org/): An end-to-end open source platform for machine learning.
- [PyTorch](https://pytorch.org/): An open source machine learning framework that accelerates the path from research prototyping to production deployment.
- [Keras](https://keras.io/): A high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.
- [Theano](http://deeplearning.net/software/theano/): A Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.
- [Caffe](https://caffe.berkeleyvision.org/): A deep learning framework made with expression, speed, and modularity in mind.
- [CNTK](https://github.com/microsoft/CNTK): Microsoft's Computational Network Toolkit (CNTK) is a library to create large-scale neural networks.
- [MXNet](https://mxnet.apache.org/): A deep learning framework designed for both efficiency and flexibility.
- [Chainer](https://chainer.org/): A Python-based deep learning framework aiming at flexibility.
- [PaddlePaddle](https://www.paddlepaddle.org.cn/): PArallel Distributed Deep LEarning: Machine learning framework supporting dynamic neural networks.
- [JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs.
- [ONNX](https://onnx.ai/): Open Neural Network Exchange (ONNX) provides an open source format for AI models.

## Generative Models

- [GPT-3](https://openai.com/research/gpt-3/): GPT-3 by OpenAI is a state-of-the-art autoregressive language model that uses deep learning to produce human-like text.
- [BERT](https://github.com/google-research/bert): BERT (Bidirectional Encoder Representations from Transformers) is a method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.
- [T5](https://github.com/google-research/text-to-text-transfer-transformer): T5 (Text-to-Text Transfer Transformer) is a model that aims to explore the limits of transfer learning in NLP by converting every language problem into a text-to-text format.
- [XLNet](https://github.com/zihangdai/xlnet): XLNet is a generalized autoregressive pretraining method that outperforms BERT on several NLP tasks.
- [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta): RoBERTa is a robustly optimized BERT pretraining approach by Facebook AI.
- [ALBERT](https://github.com/google-research/albert): ALBERT is a lite BERT for self-supervised learning of language representations.
- [ELECTRA](https://github.com/google-research/electra): ELECTRA is a new method for self-supervised language representation learning.
- [Transformer-XL](https://github.com/kimiyoung/transformer-xl): Transformer-XL is a transformer model with longer-term dependency.
- [CTRL](https://github.com/salesforce/ctrl): CTRL (Conditional Transformer Language Model) is a transformer model that can control attributes of the generated text.
- [DALL-E](https://openai.com/research/dall-e/): DALL-E by OpenAI generates images from textual descriptions.
- [CLIP](https://openai.com/research/clip/): CLIP (Contrastive Language–Image Pretraining) connects vision and language with transformers.

## Text Generation Libraries

- [Hugging Face Transformers](https://github.com/huggingface/transformers): Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, and more in 100+ languages.
- [GPT-2-simple](https://github.com/minimaxir/gpt-2-simple): A simple Python package that wraps existing model fine-tuning and generation scripts for OpenAI's GPT-2 text generation model.
- [Texar](https://github.com/asyml/texar): A general-purpose text generation toolkit in TensorFlow that emphasizes modularity and composability, making it easy to use and customize.
- [Fairseq](https://github.com/pytorch/fairseq): A general-purpose sequence-to-sequence library for PyTorch developed by Facebook AI.
- [AllenNLP](https://allennlp.org/): An open-source NLP research library, built on PyTorch.
- [DeepSpeed](https://www.deepspeed.ai/): A deep learning optimization library that makes distributed training easy, efficient, and effective.
- [PyText](https://pytext-pytext.readthedocs.io/en/latest/): A deep-learning based NLP modeling framework built on PyTorch.
- [Spacy](https://spacy.io/): A library for advanced Natural Language Processing in Python and Cython.
- [NLTK](https://www.nltk.org/): A leading platform for building Python programs to work with human language data.
- [Gensim](https://radimrehurek.com/gensim/): Gensim is a Python library for topic modelling, document indexing and similarity retrieval with large corpora.
- [FastText](https://fasttext.cc/): A library for efficient learning of word representations and sentence classification.

## Image Generation Libraries

- [StyleGAN](https://github.com/NVlabs/stylegan): StyleGAN is a generative adversarial network (GAN) introduced by Nvidia researchers in December 2018, and open sourced in February 2019.
- [BigGAN](https://github.com/ajbrock/BigGAN-PyTorch): BigGAN is a GAN that uses large-scale GAN training to generate high-fidelity natural images.
- [CycleGAN](https://github.com/junyanz/CycleGAN): CycleGAN is a technique for training unsupervised image translation models via the GAN architecture using unpaired collections of images from two different domains.
- [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): Pix2Pix is a method for training a GAN to map from input images to output images.
- [DCGAN](https://github.com/Newmu/dcgan_code): DCGAN is a direct extension to the GAN, except that it explicitly uses convolutional and convolutional-transpose layers in the discriminator and generator.
- [ProGAN](https://github.com/tkarras/progressive_growing_of_gans): ProGAN uses a progressive training method which grows the GAN from low resolution to high resolution.
- [SRGAN](https://github.com/tensorlayer/srgan): SRGAN provides a generator network for upscaling low-resolution images to high-resolution images.
- [Neural Style Transfer](https://github.com/leongatys/PytorchNeuralStyleTransfer): An algorithm that takes as input a content image and a style image and returns the content image re-imagined in the style of the style image.
- [DeepArt](https://deepart.io/): Turns your photos into art using different art styles.
- [DeepDream](https://github.com/google/deepdream): A technique that uses a convolutional neural network to find and enhance patterns in images.
- [VQ-VAE-2](https://github.com/rosinality/vq-vae-2-pytorch): A PyTorch implementation of VQ-VAE-2, a generative model that produces high quality images.

## Music Generation Libraries

- [Magenta](https://github.com/magenta/magenta): Magenta is a research project exploring the role of machine learning in the process of creating art and music.
- [OpenAI MuseNet](https://openai.com/research/musenet/): MuseNet is a deep learning model developed by OpenAI that can generate 4-minute musical compositions with 10 different instruments, and can combine styles from country to Mozart to the Beatles.
- [Jukin Media](https://www.jukinmedia.com/): Jukin Media is a global entertainment company that identifies, acquires and licenses the most compelling user-generated video content.
- [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio): WaveNet is a deep generative model of raw audio waveforms developed by DeepMind.
- [MelNet](https://github.com/ybayle/awesome-deep-learning-music#melnet): MelNet is a generative model for music that can generate high-quality, diverse and highly expressive melodies.
- [OpenAI JukinBox](https://openai.com/research/jukebox/): Jukebox is a neural net that generates music, including rudimentary singing, as raw audio in a variety of genres and artist styles.
- [FlowSynth](https://flowsynth.io/): FlowSynth is a tool for creating, modifying, and synthesizing audio using a flow-based programming language.
- [MusicVAE](https://magenta.tensorflow.org/music-vae): MusicVAE is a machine learning model that lets you easily generate and explore musical notes and rhythms.
- [PerformanceRNN](https://magenta.tensorflow.org/performance-rnn): PerformanceRNN is an LSTM-based recurrent neural network designed to model polyphonic music with expressive timing and dynamics.
- [NSynth](https://magenta.tensorflow.org/nsynth): NSynth is a machine learning algorithm that uses deep neural networks to learn and reproduce the sounds of different musical instruments.
- [DeepJ](https://github.com/olofmogren/deepj): DeepJ is a model for style-specific music generation.

These tools form the backbone of many generative AI projects, and understanding how to use them effectively can greatly enhance your productivity and the quality of your results.

## An Introduction of NeuralCoref and Coreference

NeuralCoref is a pipeline extension for spaCy 2.1+ which annotates and resolves coreference clusters using a neural network. NeuralCoref is production-ready, integrated in spaCy's NLP pipeline and extensible to new training datasets.

For a brief introduction to coreference resolution and NeuralCoref, please refer to our blog post. NeuralCoref is written in Python/Cython and comes with a pre-trained statistical model for English only.

## Training Environment Configuration

NeuralCoref uses pytorch to build the neuron network, which is defined in [the train folder](https://github.com/huggingface/neuralcoref/tree/master/neuralcoref/train). It requires the specified range of versions of python, pytorch, and spaCy. In theory python with version >=3.6, spaCy with version <3.0.0, >=2.1.0, and torch<1.4.0, >=1.3.0 should be good to train the model. Below I list the modules I installed for traininig.

```bash
pip install torch==1.3.1 torchvision==0.4.0

pip install spacy==2.3.0

python -m spacy download zh_core_web_sm

pip install tensorboardX
```

For NeuralCoref installation, you can either install the module using `pip` or clone [the GitHub repo](https://github.com/huggingface/neuralcoref) directly. You may encounter errors like **spacy.strings.StringStore size changed**, in which case an re-installation may help:

```bash
pip install neuralcoref --no-binary neuralcoref
```

When I was training the modle I needed to use this `pip` command to install NeuralCoref. In order to make the whole training environment clean, you can consider creating a python virtual environment for it.

## Training Data Preparation

Now we have all the modules installed, before starting the training process, we need to make sure that the training data is ready as well. [NeuralCoref's training document](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/training.md) explains the data preparation clearly. In general you need to download [OntoNotes 5.0 dataset](https://catalog.ldc.upenn.edu/LDC2013T19) and [CoNLL-2012 skeleton files](https://cemantix.org/conll/2012/data.html), then you can use the script [compile_coref_data.sh](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/conll_processing_script/compile_coref_data.sh) to assemble files in Chinese. If you have computer performance issues like I do, you might consider trimming files to reduce the stress of training.

After collecting the training data, the next step is the word embedding. The word embeddings are stored in [the weights folder](https://github.com/huggingface/neuralcoref/tree/master/neuralcoref/train/weights), but both `*_word_embeddings.npy` and `*_word_vocabulary.txt` are for English words, which means we have to create our own word vectors for Chinese. There are many Chinese word vectors available on the Internet, like [fastText](https://fasttext.cc/docs/en/crawl-vectors.html), [Chinese-Word-Vectors](https://github.com/Embedding/Chinese-Word-Vectors), and [Tencent AI Lab Embedding](https://ai.tencent.com/ailab/nlp/en/embedding.html).

For those who are curious about how to convert txt embedding file to `*_word_embeddings.npy` and `*_word_vocabulary.txt`. Here is [the example code](https://gist.github.com/erickrf/e54cd0f3d917ec61b3ae758a5e47b883).

## Training Process Explanation and Possible Failures

Let's take a look at the training process. First we need to make sure the correct spaCy model is loaded, the default model in the train folder is [English](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/conllparser.py#L752), we can change it `zh_core_web_sm` for Chinese:

```python
        print("🌋 Loading spacy model")

        if model is None:
            model_options = ["zh_core_web_sm", "zh_core_web_trf"]
            for model_option in model_options:
                if not model:
                    try:
                        spacy.info(model_option)
                        model = model_option
                        print("Loading model", model_option)
                    except:
                        print("Could not detect model", model_option)
            if not model:
                print("Could not detect any suitable Chinese model")
                return
        else:
            spacy.info(model)
            print("Loading model", model)
```

We need to update [`document.py`](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/document.py#L962) as well:

```python
    print("🌋 Loading spacy model")
    try:
        spacy.info("zh_core_web_sm")
        model = "zh_core_web_sm"
    except IOError:
        print("No spacy 2 model detected, using spacy 'zh_core_web_trf' model")
        spacy.info("zh_core_web_trf")
        model = "zh_core_web_trf"
```

Then we need to adjust the input size of the neural network because different word embeddings may have different dimensions. The size configuration is located in [`utils.py`](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/utils.py#L14) as global variables:

```python
SIZE_SPAN = 250  # size of the span vector (averaged word embeddings)
SIZE_WORD = 8  # number of words in a mention (tuned embeddings)
SIZE_EMBEDDING = 50  # size of the words embeddings
SIZE_FP = 70  # number of features for a pair of mention
...
```

NeuralCoref's default Engish word vector's dimension is 50. If I use fastText's word vectors, in which case the Chinese word vector's dimension becomes 300, the size configuration will be:

```python
SIZE_SPAN = 1500  # size of the span vector (averaged word embeddings)
SIZE_WORD = 8  # number of words in a mention (tuned embeddings)
SIZE_EMBEDDING = 300  # size of the words embeddings
SIZE_FP = 70  # number of features for a pair of mention
```

With all the data preparation and code modification we have accomplished, we can start the training process. The training process consists of 2 steps, the first step is to preprocess the data and the second step is to train and evaluate the neural network.

Preprocessing the data is to translate the conll files into numpy files so that the neural network can understand, we need to perform it for train, test, and dev [datasets](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/train/training.md#prepare-the-data).

```bash
python -m neuralcoref.train.conllparser --path ./$path_to_data_directory
```

Then we can train the model using the following commands:

```bash
python -m neuralcoref.train.learn --train ./data/train/ --eval ./data/dev/
```

The neural network has 5 layers:

```python
ReLu >> ReLu >> ReLu >> Affine >> Affine
```

The weights and bias of each training epoch are stroed in [the checkpoints folder](https://github.com/huggingface/neuralcoref/tree/master/neuralcoref/train/checkpoints), `learn.py` also periodcally stores best weights and bias settings so far into the folder. This feauture also allows us to start the training from a specific checkpoint, which is very useful.

```bash
python -m neuralcoref.train.learn --checkpoint_file ./checkpoints/Apr06_10-04-37_example_best_modelallpairs --train ./data/train/ --eval ./data/dev/
```

Based on my observation, the files in the train folder of NeuralCoref downloaded from `pip` may have the wrong module path. For example in `learn.py` it may have:

```python
from neuralcoref.utils import SIZE_EMBEDDING
from neuralcoref.evaluator import ConllEvaluator
```

In this situation you need to modify it to:

```python
from neuralcoref.train.utils import SIZE_EMBEDDING
from neuralcoref.train.evaluator import ConllEvaluator
```

## Integrate Trained Model with NeuralCoref

Once you are satisfied with the results of the training, you can try to integrate the trained neural network with NeuralCoref. Based on [NeuralCoref's README](https://github.com/huggingface/neuralcoref#internals-and-model), the first time you import NeuralCoref in python, it will download the weights of the neural network model in a cache folder. It means NeuralCoref uses default parameters to set up its model. How do we configure the model with our trained weights and bias? There is no relavent documents so far, so in this section I would like to talk about how I integrated my trained model into NeuralCoref module.

The most important thing is to load trained weights and bias into NeuralCoref's model, we can implement it by loading the best checkpoint file:

```python
import torch

with Model.define_operators({'**': clone, '>>': chain}):
            single_model = ReLu(h1, SIZE_SINGLE_IN) >> ReLu(h2, h1) >> ReLu(h3, h2) >> Affine(1, h3) >> Affine(1, 1)
            pairs_model = ReLu(h1, SIZE_PAIR_IN) >> ReLu(h2, h1) >> ReLu(h3, h2) >> Affine(1, h3) >> Affine(1, 1)
            
        tm = torch.load("best_modelallpairs")

        pairs_model._layers[0].W = tm["pair_top.0.weight"].cpu().numpy()
        pairs_model._layers[0].b = tm["pair_top.0.bias"].cpu().numpy()
        ...

        single_model._layers[0].W = tm["single_top.0.weight"].cpu().numpy()
        single_model._layers[0].b = tm["single_top.0.bias"].cpu().numpy()
        ...
```

[This thread](https://github.com/huggingface/neuralcoref/issues/257) has a very good discussion about this topic.

You can either put the code example to [`nerucoref.pyx`](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/neuralcoref.pyx#L497) directly or use it separately and replace NeuralCoref's model later.

Since we use Chinese word vectors to train the model and NeuralCoref loads the English word vectors from the cache folder by default, we need to take a look at the word vector. We can modify `nerucoref.pyx` to load the Chinese word vectors.

```python
self.static_vectors = numpy.load("static_word_embeddings.npy")
```

Then you need to modify [`get_word_embedding`](https://github.com/huggingface/neuralcoref/blob/master/neuralcoref/neuralcoref.pyx#L878) as well to make it use the Chinese word vectors.

A better alternative is to pack the Chinese word vector as a spaCy model and store it into the cache folder so that NeuralCoref can fetch and load it directly.

```bash
gzip word2vec.txt
python -m spacy init-model zh ./data/spacy.word2vec.model --vectors-loc word2vec.txt.gz
```

```python
# load model
nlp = spacy.load(model)
# store it into cache folder
nlp.vocab.vectors.to_disk(path)
```

## Examples of Using NeuralCoref with Trained Model

With the trained model integrated with NeuralCoref, we can strat using it!

```python
import spacy
nlp = spacy.load("zh_model")
import neuralcoref
neuralcoref.add_to_pipe(nlp)
doc = nlp('大家爱我。他们对我好。')
doc._.has_coref
doc._.coref_clusters
```

Since the hardware limitation of my laptop, I only trained the model with a few samples and the result of my Chinese NeuralCoref is not very good. I hope you can train your more powerful CPU/GPU with more data and the result will be excellent!

## Discussion and Questions

If you have any question with this tutorial, please feel free to post it to the [GitHub issues page](https://github.com/cxl07/cxl07.github.io/issues) and we can talk there!

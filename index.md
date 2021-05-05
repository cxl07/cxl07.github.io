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

## Integrate Trained Model with NeuralCoref Library

## Examples of Using NeuralCoref with Trained Model

## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/cxl07/cxl07.github.io/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/cxl07/cxl07.github.io/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

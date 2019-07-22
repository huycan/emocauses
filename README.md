# Emotion causal relationship in conversation

## Intro

This repo contains public datasets and associated source codes to run experiments for my Honour thesis project at Monash.

Notes of methods, experiments, and results are also recorded along the way.

## Setup

### (Optional) Intel Math Kernel Library (MKL)

**If you decide to use this lib, it should be installed first, as all subsequent libs install might reference this** 

This boosts linear algebra computation for both R and Python numerical computing packages.

- Installation guide
<http://dirk.eddelbuettel.com/blog/2018/04/15/#018_mkl_for_debian_ubuntu>

- Reference this list for latest MLK build
<https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo>

- Installed version is **intel-mkl-64bit-2019.4-070**

### R

- Add CRAN repo to Ubuntu, then install latest R package
<https://cran.rstudio.com/>

- Installed version is **3.6.1**

#### Inference Algorithms

Please see [R code](causal.Rmd) for further package installation

### Python

- Install from source
    - Download latest release from <https://www.python.org/downloads/>. Installed version is **3.8.0**
    - (Optional) Remove existing python3 installation, **be careful** to not delete Ubuntu system python3
    - Build and install from source, **be careful** to not override Ubuntu system python3
    <https://linuxize.com/post/how-to-install-python-3-7-on-ubuntu-18-04/>

- Best to build new release and use **Python3.8** directly. As the default Python3 installed via Ubuntu apt has a few custom scripts important for the system
<https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-7-on-ubuntu-18-10/>

- Setup a Python venv
<https://docs.python.org/3/library/venv.html>

```shell script
python3.8 -m venv .venv
source .venv/bin/activate
```

#### SpaCy

- Install SpaCy via pip
<https://spacy.io/usage>

```shell script
source .venv/bin/activate
pip install -U spacy
pip install -U spacy-lookups-data
python -m spacy download en_core_web_sm
```

#### Python Packages

cython, numpy, scipy, scikit-learn, spaCy, pandas

#### (Optional) Build numpy with MKL-enabled

This post details how to compile numpy from source with MKL-enabled.
<https://www.elliottforney.com/blog/npspmkl/> 

pandas, scipy, and scikit-learn use numpy.

If you decide to go this route for Python performance boost, remove existing installation of these libraries first, install numpy with MKL, then re-build and install them from source

### Dialogue Act Tagger

The folder tagger is forked from this project. It is a Git submodule reference the forked project. The fork contains my own change to speed up the project and adapt to my research.

[Original](https://github.com/ColingPaper2018/DialogueAct-Tagger/commit/175a57f6c32475efbc01009afc6f5a0180b2d180)

[Fork](https://github.com/ColingPaper2018/DialogueAct-Tagger)

Install scikit-learn, spaCy

Install spaCy en-core model

Since the tagger folder is a child folder, it shares the same set of Python libraries as the main repo

## Datasets

### Dyadic Conversation

MELD Friends

### Dialogue Act Tagger

AMI
Oasis BT (official license expires in 2009, dataset download link is still accessible)
Switchboard
Maptask

Links for dataset download is in the original repo <https://github.com/ColingPaper2018/DialogueAct-Tagger>

## Steps

1) Download 3 datasets for tagger
2) Train tagger model as instructed in the author's original repo
3) Run nlp script to model conversation features
4) Run R script for inference

## References

[MELD dataset](https://affective-meld.github.io/)
Poria, S., Hazarika, D., Majumder, N., Naik, G., Cambria, E., & Mihalcea, R. (2018). Meld: A multimodal multi-party dataset for emotion recognition in conversations. ​ACL (57).

[ISO independent dialogue tagger](https://github.com/ColingPaper2018/DialogueAct-Tagger)
Mezza, S., Cervone, A., Tortoreto, G., Stepanov, E. A., & Riccardi, G. (2018). ISO-standard domain-independent dialogue act tagging for conversational agents. ​ COLING, (27) ​ .

[pcalg](https://cran.r-project.org/package=pcalg)
Kalisch, M., Hauser, A., Maechler, M., Colombo, D., Entner, D., Hoyer, P., ..., Eigenmann, M. (2019). pcalg: Methods for Graphical Models and Causal Inference (version 2.6-7) [Computer Software]. Retrieved from ​<https://cran.r-project.org/package=pcalg>

Causal inference theory behind PC and FCI 
Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, prediction, and search. Adaptive computation and machine learning.

[InvariantCausalPrediction](https://cran.r-project.org/package=InvariantCausalPrediction)
Meinshausen, N. (2018). InvariantCausalPrediction: Invariant Causal Prediction (version 0.7-2) [Computer Software]. Retrieved from <https://cran.r-project.org/package=InvariantCausalPrediction>

Theory behind InvariantCausalInference
Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference by using invariant prediction: identification and confidence intervals. ​ Journal of the Royal Statistical Society: Series B (Statistical Methodology), 78 ​ (5), 947-1012.

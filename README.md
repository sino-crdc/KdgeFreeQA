# KdgeFreeQA
Knowledge-guided free question-answering system, aka, KQA (by [KQA Team@CRI](https://github.com/orgs/sino-crdc/teams/kqa_team)).

## Initial Algorithm to be extended
[Generating Well-Formed Answers by Machine Reading with Stochastic Selector Networks](https://ojs.aaai.org//index.php/AAAI/article/view/6238) \
**Abstract** : Question answering (QA) based on machine reading comprehension has been a recent surge in popularity, yet most work has focused on extractive methods.We instead address a more challenging QA problem of generating a well-formed answer by reading and summarizing the paragraph for a given question. For the generative QA task, we introduce a new neural architecture, LatentQA, in which a novel stochastic selector network composes a well-formed answer with words selected from the question, the paragraph and the global vocabulary, based on a sequence of discrete latent variables. Bayesian inference for the latent variables is performed to train the LatentQA model. The experiments on public datasets of natural answer generation confirm the effectiveness of LatentQA in generating high-quality well-formed answers.

## Dataset
Our original dataset is collected from [CAIL 2021](http://cail.cipsc.org.cn/). \
After a simple adjustment, the dataset is saved at bhpan (Beihang Cloud Disk).\
See data/ for the dataset adjustment.

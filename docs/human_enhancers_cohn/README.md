# Human Enhancers

Human enhancers as described in the paper [Enhancer Identification using Transfer and
Adversarial Deep Learning of DNA Sequences](https://www.biorxiv.org/content/biorxiv/early/2018/02/14/264200.full.pdf). The paper talks about enhancers in multiple organisms but we have imported only human enhancers. More precisely, we are only using the folder "Human" of [Enhancers_vs_negative.tgz](http://www.cs.huji.ac.il/~tommy//enhancer_CNN/Enhancers_vs_negative.tgz) file referenced from the authors' [GitHub repository](https://github.com/cohnDikla/enhancer_CNN).

The data have been imported with `seq2loc` utility. Totally, 13895 out of 14000 human enhancers and 13896 out of 140000 negative controls have been found in the human genome reference and included.
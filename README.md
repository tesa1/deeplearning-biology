# deeplearning in genomics and epigenomics

This is a list of implementations of deep learning methods to biology. 
Originally forked from https://github.com/hussius/deeplearning-biology

## Reviews <a name="reviews"></a>

These are not implementations as such, but contain useful pointers. Because review papers in this field are more time-sensitive, I have added the month of journal publication. Note that the original preprint may in some cases have been available online long before the published version.

**(2019-12) Deep learning of pharmacogenomics resources: moving towards precision oncology** [[Briefings in Bioinformatics](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbz144/5669856#186956080)]

**(2019-04) Deep learning: new computational modelling techniques for genomics** [[Nature Reviews Genetics paper](https://www.nature.com/articles/s41576-019-0122-6)]

This is a very nice conceptual review of how deep learning can be used in genomics. It explains how convolutional networks, recurrent networks, graph convolutional networks, autoencoders and GANs work. It also explains useful concepts like multi-modal learning, transfer learning, and model explainability.

**(2019-01) A guide to deep learning in healthcare** [[Nature Medicine paper](https://www.nature.com/articles/s41591-018-0316-z)]

From the abstract: "Here we present deep-learning techniques for healthcare, centering our discussion on deep learning in computer vision, natural language processing, reinforcement learning, and generalized methods. We describe how these computational techniques can impact a few key areas of medicine and explore how to build end-to-end systems. Our discussion of computer vision focuses largely on medical imaging, and we describe the application of natural language processing to domains such as electronic health record data. Similarly, reinforcement learning is discussed in the context of robotic-assisted surgery, and generalized deep-learning methods for genomics are reviewed."

**(2018-11) A primer on deep learning in genomics** [[Nature Genetics paper](https://www.nature.com/articles/s41588-018-0295-5)][[Colaboratory notebook with tutorial](https://colab.research.google.com/drive/17E4h5aAOioh5DiTo7MZg4hpL6Z_0FyWr)]

This review, which features yours truly as one of its co-authors, is billed as a 'primer' which means it tries to help genomics researchers get started with deep learning. We tried to accomplish this by highlighting many practical issues such as tooling (not only deep learning libraries but also GPU cloud platforms, model zoos and online courses), defining your deep learning problem, explainability and troubleshooting. We also made a tutorial on Colaboratory that shows how to set up and run a simple convolutional network model for learning binding motifs, and how to inspect the model's predictions after it has been trained.


## Genomics <a name="genomics"></a>

This category is divided into several subfields.

### Variant calling <a name='genomics_variant-calling'></a>

**DeepVariant** [[paper](https://www.nature.com/articles/nbt.4235)]

This preprint from Google originally came out in late 2016 but it got the most publicity about a year later when the code was made public and press releases started appearing. The Google researchers approached a well-studied problem, variant calling from DNA sequencing data (where the aim is to correctly identify variations from the reference genome in an individual's DNA, e.g. mutations or polymorphisms) using a counter-intuitive but clever approach. Instead of using the nucleotides in the sequenced DNA fragments directly (in the form of the symbols A, C, G, T), they first converted the sequences into images and then applied convolutional neural networks to these images (which represent "pile-ups" or DNA sequences; stacks of aligned sequences.) This turned out to be a very effective way to call variants as proven by both Google's own and independent benchmarks.


**Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk** [[paper](https://www.nature.com/articles/s41588-018-0160-6)]

The authors use a two-step model to predict the effect of genetic variants on gene expression. In the first step, the authors trained a convolutional neural network to model the 2002 epigenetic marks collected in ENCODE and ROADMAP consortium. In the second step, the authors trained a tissue-specific regularized linear model on the cis-regulatory region of the gene that is encoded by the first step convolutional neural network model. Then the effect of the variants on tissue-specific gene is calculated by the decrease in predicted gene expression through *in silico* mutagenesis.


### Predicting enhancers and regulatory regions <a name='genomics_enhancers'></a>

**(2016) DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences** [[paper](https://academic.oup.com/nar/article/44/11/e107/2468300)]

Made for predicting the function of non-protein coding DNA sequence. Uses a convolution layer to capture regulatory motifs (i e single DNA snippets that control the expression of genes, for instance), and a recurrent layer (of the LSTM type) to try to discover a “grammar” for how these single motifs work together. Based on Keras/Theano.

**Basset – learning the regulatory code of the accessible genome with deep convolutional neural networks** [[github](https://github.com/davek44/Basset)][[paper](https://pubmed.ncbi.nlm.nih.gov/27197224/)]

Based on Torch, this package focuses on predicting the accessibility (or “openness”) of the chromatin – the physical packaging of the genetic information (DNA+associated proteins). This can exist in more condensed or relaxed states in different cell types, which is partly influenced by the DNA sequence (not completely, because then it would not differ from cell to cell.)

**Basenji – Sequential regulatory activity prediction across chromosomes with convolutional neural networks** [[paper](https://genome.cshlp.org/content/28/5/739.long)]

A follow-up project to Basset, this Tensorflow-based model uses both standard and dilated convolutions to model regulatory signals and gene expression (in the form of CAGE tag density) in many different cell types. Notably, the underlying model has been brought into Google's Tensor2Tensor repository (see "github2" link above), which collects many models in image and speech recognition, machine translation, text classification etc. However, at the time of writing the Tensor2Tensor model seems not quite mature for easy use, so it is probably better to use the dedicated Basenji repo ("github1") for now. 

**DeepSEA – Predicting effects of noncoding variants with deep learning–based sequence model** [[paper](http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html)]

Like the packages above, this one also models chromatin accessibility as well as the binding of certain proteins (transcription factors) to DNA and the presence of so-called histone marks that are associated with changes in accessibility. This piece of software seems to focus a bit more explicitly than the others on predicting how single-nucleotide mutations affect the chromatin structure. Published in a high-profile journal (Nature Methods).

**DeepBind – Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning** [[paper](http://www.nature.com/nbt/journal/v33/n8/full/nbt.3300.html)]

This is from the group of Brendan Frey in Toronto, and the authors are also involved in the company Deep Genomics. DeepBind focuses on predicting the binding specificities of DNA-binding or RNA-binding proteins, based on experiments such as ChIP-seq, ChIP-chip, RIP-seq,  protein-binding microarrays, and HT-SELEX. Published in a high-profile journal (Nature Biotechnology.)

**DeeperBind - Enhancing Prediction of Sequence Specificities of DNA Binding Proteins** [[preprint](https://arxiv.org/pdf/1611.05777.pdf)]

This is an attempt to improve on DeepBind by adding a recurrent sequence learning module (LSTM) after the convolutional layer(s). In this way, the authors propose to capture a positional dimension that is lost in the pooling step in the original DeepBind design. They claim that benchmarking shows that this architecture leads to superior performance compared to previous work.

**DeepMotif - Visualizing Genomic Sequence Classifications** [[paper](https://arxiv.org/abs/1605.01133)]

This is also about learning and predicting binding specificities of proteins to certain DNA patterns or "motifs". However, this paper makes use of a combination of convolutional layers and [highway networks](https://arxiv.org/pdf/1505.00387v2.pdf), with more layers than the DeepBind network. The authors also show how a learned classifier can generate typical DNA motifs by input optimization; applying back-propagation with all the weights held constant in order to find an input pattern that maximally activates the appropriate output node in the network.

**Convolutional Neural Network Architectures for Predicting DNA-Protein Binding** [[paper](http://bioinformatics.oxfordjournals.org/content/32/12/i121.full)]

This work describes a systematic exploration of convolutional neural network (CNN) architectures for DNA-protein binding. It concludes that the convolutional kernels are very important for the success of the networks on motif-based tasks. Interestingly, the authors have provided a Dockerized implementation of DeepBind from the Frey lab (see above) and also provide EC2-laucher scripts and code for comparing different GPU enabled models programmed in Caffe.

**PEDLA: predicting enhancers with a deep learning-based algorithmic framework** [[paper](http://biorxiv.org/content/early/2016/01/07/036129)]

This package is for predicting enhancers (stretches of DNA that can enhance the expression of a gene under certain conditions or in a certain kind of cell, often working at a distance from the gene itself) based on heterogeneous data from (e.g.) the ENCODE project, using 1,114 features altogether.

**DEEP: a general computational framework for predicting enhancers** [[paper](http://nar.oxfordjournals.org/content/early/2014/11/05/nar.gku1058.full)][[code](http://cbrc.kaust.edu.sa/deep/)]

An ensemble prediction method for enhancers.

**Genome-Wide Prediction of cis-Regulatory Regions Using Supervised Deep Learning Methods** (and several other papers applying various kinds of deep networks to regulatory region prediction) (one [[paper](http://biorxiv.org/content/early/2016/02/28/041616)] out of several)

Wyeth Wasserman’s group have made a kind of [toolkit](https://github.com/yifeng-li/DECRES) (based on the Theano tutorials) for applying different kinds of deep learning architectures to cis-regulatory element (DNA stretches that can modulate the expression of a nearby gene) prediction. They use a specific “feature selection layer” in their nets to restrict the number of features in the models. This is implemented as an additional sparse one-to-one linear layer between the input layer and the first hidden layer of a multi-layer perceptron.

**FIDDLE: An integrative deep learning framework for functional genomic data inference** [[paper](http://biorxiv.org/content/early/2016/10/17/081380)]

The group predicted transcription start site and regulatory regions but claims this solution could be easily generalized and predict other features too. FIDDLE stands for Flexible Integration of Data with Deep LEarning. The idea is to model several genomic signals jointly using convolutional networks. This could be for example DNase-seq, ATAC-seq, ChIP-seq, TSS-seq, maybe RNA-seq signals (as in .wig files with one value per base in the genome).

**Deep Learning Of The Regulatory Grammar Of Yeast 5′ Untranslated Regions From 500,000 Random Sequences** [[paper](http://genome.cshlp.org/content/27/12/2015)]

This is a CNN model that attempts to predict protein expression from the DNA sequence in a specific type of genomic region called 5' UTR (five-prime untranslated region). The model is built in Keras and a nice touch by the authors is that they optimized the parameters using hyperopt, which is also shown in one of the Jupyter notebooks that comes along with the paper. The results look promising and easily reproducible, judging from my own trial.

**Modeling Enhancer-Promoter Interactions with Attention-Based Neural Networks** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/14/219667)]

The concept of attention in (recurrent) neural networks has become quite popular recently, not least because it has been used to great effect in machine translation models. This paper proposes an attention-based model for getting at the interactions between enhancer sequences and promoter sequences.

**Predicting Transcription Factor Binding Sites with Convolutional Kernel Networks** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/10/217257)]

This paper uses a hybrid of CNNs (to learn good representations) and kernel methods (to learn good prediction functions) to predict transcription factor binding sites.

**Predicting DNA accessibility in the pan-cancer tumor genome using RNA-seq, WGS, and deep learning** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/12/05/229385)]

Like Basset (above) this paper shows how to predict DNA accessibility from sequence using CNNs, but it adds the possibility to leverage RNA sequencing data from different cell types as input. In this way implicit information related to cell type can be "transferred" to the accessibility prediction task.

**Deep learning at base-resolution reveals motif
syntax of the cis-regulatory code** [[bioRxiv preprint](https://www.biorxiv.org/content/biorxiv/early/2019/08/21/737981.full.pdf)]

Here, a CNN with dilated convolutions is used to learn how different transcription factor binding motifs cooperate. This is the "motif syntax" mentioned in the title. The neural network is trained to predict the signal from a basepair-resolution ChIP assay (ChIP-nexus) and the trained network is then used to infer rules of motif cooperativity.

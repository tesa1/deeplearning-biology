# deeplearning-non-coding

This is a list of implementations of deep learning methods to biology, originally published on [Follow the Data](https://followthedata.wordpress.com/). There is a slant towards genomics because that's the subfield that I follow most closely.

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

**DeepVariant** [[github](https://github.com/google/deepvariant)][[preprint](https://www.biorxiv.org/content/early/2016/12/21/092890)]

This preprint from Google originally came out in late 2016 but it got the most publicity about a year later when the code was made public and press releases started appearing. The Google researchers approached a well-studied problem, variant calling from DNA sequencing data (where the aim is to correctly identify variations from the reference genome in an individual's DNA, e.g. mutations or polymorphisms) using a counter-intuitive but clever approach. Instead of using the nucleotides in the sequenced DNA fragments directly (in the form of the symbols A, C, G, T), they first converted the sequences into images and then applied convolutional neural networks to these images (which represent "pile-ups" or DNA sequences; stacks of aligned sequences.) This turned out to be a very effective way to call variants as proven by both Google's own and independent benchmarks.



**Boosting Gene Expression Clustering with System-Wide Biological Information: A Robust Autoencoder Approach** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/05/214122)]

Uses a robust autoencoder (an autoencoder with an outlier filter) to cluster gene expression profiles. 

**Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk** [[github](https://github.com/FunctionLab/ExPecto)][[paper](https://www.nature.com/articles/s41588-018-0160-6)]

The authors use a two-step model to predict the effect of genetic variants on gene expression. In the first step, the authors trained a convolutional neural network to model the 2002 epigenetic marks collected in ENCODE and ROADMAP consortium. In the second step, the authors trained a tissue-specific regularized linear model on the cis-regulatory region of the gene that is encoded by the first step convolutional neural network model. Then the effect of the variants on tissue-specific gene is calculated by the decrease in predicted gene expression through *in silico* mutagenesis.

### Imaging and gene expression <a name='imaging_expression'></a>

**Transcriptomic learning for digital pathology** [[preprint](https://www.biorxiv.org/content/biorxiv/early/2019/10/11/760173.full.pdf)]

From the abstract: "We propose a novel approach based on the integration of multiple data modes, and show that our deep learning model, HE2RNA, can be trained to systematically predict RNA-Seq profiles from whole-slide images alone, without the need for expert annotation. HE2RNA is interpretable by design, opening up new opportunities for virtual staining. In fact, it provides virtual spatialization of gene expression,as validated by double-staining on an independent dataset. Moreover, the transcriptomic representation learned by HE2RNA can be transferred to improve predictive performance for other tasks, particularly for small datasets."

### Predicting enhancers and regulatory regions <a name='genomics_enhancers'></a>

Here the inputs are typically “raw” DNA sequence, and convolutional networks (or layers) are often used to learn regularities within the sequence. Hat tip to [Melissa Gymrek](http://melissagymrek.com/science/2015/12/01/unlocking-noncoding-variation.html) for pointing out some of these.

**DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences** [[github](https://github.com/uci-cbcl/DanQ)][[gitxiv](http://gitxiv.com/posts/aqrWwLoyg75jqNAYX/danq-a-hybrid-convolutional-and-recurrent-deep-neural)]

Made for predicting the function of non-protein coding DNA sequence. Uses a convolution layer to capture regulatory motifs (i e single DNA snippets that control the expression of genes, for instance), and a recurrent layer (of the LSTM type) to try to discover a “grammar” for how these single motifs work together. Based on Keras/Theano.

**Basset – learning the regulatory code of the accessible genome with deep convolutional neural networks** [[github](https://github.com/davek44/Basset)][[gitxiv](http://gitxiv.com/posts/fhET6G7gnBrGS8S9u/basset-learning-the-regulatory-code-of-the-accessible-genome)]

Based on Torch, this package focuses on predicting the accessibility (or “openness”) of the chromatin – the physical packaging of the genetic information (DNA+associated proteins). This can exist in more condensed or relaxed states in different cell types, which is partly influenced by the DNA sequence (not completely, because then it would not differ from cell to cell.)

**Basenji – Sequential regulatory activity prediction across chromosomes with convolutional neural networks** [[github1](https://www.github.com/calico/basenji)][[github2](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/gene_expression.py)][[biorxiv](https://www.biorxiv.org/content/early/2017/07/10/161851)]

A follow-up project to Basset, this Tensorflow-based model uses both standard and dilated convolutions to model regulatory signals and gene expression (in the form of CAGE tag density) in many different cell types. Notably, the underlying model has been brought into Google's Tensor2Tensor repository (see "github2" link above), which collects many models in image and speech recognition, machine translation, text classification etc. However, at the time of writing the Tensor2Tensor model seems not quite mature for easy use, so it is probably better to use the dedicated Basenji repo ("github1") for now. 

**DeepSEA – Predicting effects of noncoding variants with deep learning–based sequence model** [[web server](http://deepsea.princeton.edu/job/analysis/create/)][[paper](http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html)]

Like the packages above, this one also models chromatin accessibility as well as the binding of certain proteins (transcription factors) to DNA and the presence of so-called histone marks that are associated with changes in accessibility. This piece of software seems to focus a bit more explicitly than the others on predicting how single-nucleotide mutations affect the chromatin structure. Published in a high-profile journal (Nature Methods).

**DeepBind – Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning** [[code](http://tools.genes.toronto.edu/deepbind/)][[paper](http://www.nature.com/nbt/journal/v33/n8/full/nbt.3300.html)]

This is from the group of Brendan Frey in Toronto, and the authors are also involved in the company Deep Genomics. DeepBind focuses on predicting the binding specificities of DNA-binding or RNA-binding proteins, based on experiments such as ChIP-seq, ChIP-chip, RIP-seq,  protein-binding microarrays, and HT-SELEX. Published in a high-profile journal (Nature Biotechnology.)

**DeeperBind - Enhancing Prediction of Sequence Specificities of DNA Binding Proteins** [[preprint](https://arxiv.org/pdf/1611.05777.pdf)]

This is an attempt to improve on DeepBind by adding a recurrent sequence learning module (LSTM) after the convolutional layer(s). In this way, the authors propose to capture a positional dimension that is lost in the pooling step in the original DeepBind design. They claim that benchmarking shows that this architecture leads to superior performance compared to previous work.

**DeepMotif - Visualizing Genomic Sequence Classifications** [[paper](https://arxiv.org/abs/1605.01133)]

This is also about learning and predicting binding specificities of proteins to certain DNA patterns or "motifs". However, this paper makes use of a combination of convolutional layers and [highway networks](https://arxiv.org/pdf/1505.00387v2.pdf), with more layers than the DeepBind network. The authors also show how a learned classifier can generate typical DNA motifs by input optimization; applying back-propagation with all the weights held constant in order to find an input pattern that maximally activates the appropriate output node in the network.

**Convolutional Neural Network Architectures for Predicting DNA-Protein Binding** [[code](http://cnn.csail.mit.edu/)][[paper](http://bioinformatics.oxfordjournals.org/content/32/12/i121.full)]

This work describes a systematic exploration of convolutional neural network (CNN) architectures for DNA-protein binding. It concludes that the convolutional kernels are very important for the success of the networks on motif-based tasks. Interestingly, the authors have provided a Dockerized implementation of DeepBind from the Frey lab (see above) and also provide EC2-laucher scripts and code for comparing different GPU enabled models programmed in Caffe.

**PEDLA: predicting enhancers with a deep learning-based algorithmic framework** [[code](https://github.com/wenjiegroup/PEDLA)][[paper](http://biorxiv.org/content/early/2016/01/07/036129)]

This package is for predicting enhancers (stretches of DNA that can enhance the expression of a gene under certain conditions or in a certain kind of cell, often working at a distance from the gene itself) based on heterogeneous data from (e.g.) the ENCODE project, using 1,114 features altogether.

**DEEP: a general computational framework for predicting enhancers** [[paper](http://nar.oxfordjournals.org/content/early/2014/11/05/nar.gku1058.full)][[code](http://cbrc.kaust.edu.sa/deep/)]

An ensemble prediction method for enhancers.

**Genome-Wide Prediction of cis-Regulatory Regions Using Supervised Deep Learning Methods** (and several other papers applying various kinds of deep networks to regulatory region prediction) [[code](https://github.com/yifeng-li/DECRES)] (one [[paper](http://biorxiv.org/content/early/2016/02/28/041616)] out of several)

Wyeth Wasserman’s group have made a kind of [toolkit](https://github.com/yifeng-li/DECRES) (based on the Theano tutorials) for applying different kinds of deep learning architectures to cis-regulatory element (DNA stretches that can modulate the expression of a nearby gene) prediction. They use a specific “feature selection layer” in their nets to restrict the number of features in the models. This is implemented as an additional sparse one-to-one linear layer between the input layer and the first hidden layer of a multi-layer perceptron.

**FIDDLE: An integrative deep learning framework for functional genomic data inference** [[paper](http://biorxiv.org/content/early/2016/10/17/081380)][[code](https://github.com/ueser/FIDDLE)][[Youtube talk](https://www.youtube.com/watch?v=pcLTUsOm5pc&feature=youtu.be&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS&t=2411)]

The group predicted transcription start site and regulatory regions but claims this solution could be easily generalized and predict other features too. FIDDLE stands for Flexible Integration of Data with Deep LEarning. The idea (nicely explained by the author in the YouTube video above) is to model several genomic signals jointly using convolutional networks. This could be for example DNase-seq, ATAC-seq, ChIP-seq, TSS-seq, maybe RNA-seq signals (as in .wig files with one value per base in the genome).

**Deep Learning Of The Regulatory Grammar Of Yeast 5′ Untranslated Regions From 500,000 Random Sequences** [[paper](http://genome.cshlp.org/content/27/12/2015)][[code](http://genome.cshlp.org/content/suppl/2017/11/02/gr.224964.117.DC1/Supplemental_code.tar.gz)]

This is a CNN model that attempts to predict protein expression from the DNA sequence in a specific type of genomic region called 5' UTR (five-prime untranslated region). The model is built in Keras and a nice touch by the authors is that they optimized the parameters using hyperopt, which is also shown in one of the Jupyter notebooks that comes along with the paper. The results look promising and easily reproducible, judging from my own trial.

**Modeling Enhancer-Promoter Interactions with Attention-Based Neural Networks** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/14/219667)][[code](https://github.com/wgmao/EPIANN)]

The concept of attention in (recurrent) neural networks has become quite popular recently, not least because it has been used to great effect in machine translation models. This paper proposes an attention-based model for getting at the interactions between enhancer sequences and promoter sequences.

**Predicting Transcription Factor Binding Sites with Convolutional Kernel Networks** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/10/217257)][[code](https://gitlab.inria.fr/dchen/CKN-seq)]

This paper uses a hybrid of CNNs (to learn good representations) and kernel methods (to learn good prediction functions) to predict transcription factor binding sites.

**Predicting DNA accessibility in the pan-cancer tumor genome using RNA-seq, WGS, and deep learning** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/12/05/229385)]

Like Basset (above) this paper shows how to predict DNA accessibility from sequence using CNNs, but it adds the possibility to leverage RNA sequencing data from different cell types as input. In this way implicit information related to cell type can be "transferred" to the accessibility prediction task.

**Deep learning at base-resolution reveals motif
syntax of the cis-regulatory code** [[bioRxiv preprint](https://www.biorxiv.org/content/biorxiv/early/2019/08/21/737981.full.pdf)]

Here, a CNN with dilated convolutions is used to learn how different transcription factor binding motifs cooperate. This is the "motif syntax" mentioned in the title. The neural network is trained to predict the signal from a basepair-resolution ChIP assay (ChIP-nexus) and the trained network is then used to infer rules of motif cooperativity.

### Non-coding RNA <a name='genomics_non-coding'></a>

**DeepLNC, a long non-coding RNA prediction tool using deep neural network** [[paper](http://link.springer.com/article/10.1007%2Fs13721-016-0129-2)] [[web server](http://bioserver.iiita.ac.in/deeplnc/)]

Identification of potential long non-coding RNA molecules from DNA sequence, based on k-mer profiles.

**A Deep Recurrent Neural Network Discovers Complex Biological Rules to Decipher RNA Protein-Coding Potential** [[github](https://github.com/hendrixlab/mRNN)][[paper](https://www.biorxiv.org/content/early/2017/11/13/200758.1)] 

From the abstract: *While traditional, feature-based methods for RNA classification are limited by current scientific knowledge, deep learning methods can independently discover complex biological rules in the data de novo. We trained a gated recurrent neural network (RNN) on human messenger RNA (mRNA) and long noncoding RNA (lncRNA) sequences. Our model, mRNA RNN (mRNN), surpasses state-of-the-art methods at predicting protein-coding potential.*

### Methylation <a name='genomics_methylation'></a>

**DeepCpG - Predicting DNA methylation in single cells**
[[paper](http://dx.doi.org/10.1186/s13059-017-1189-z)]
[[code](https://github.com/cangermueller/deepcpg)]
[[docs](http://deepcpg.readthedocs.io/en/latest/)]

DeepCpG is a deep neural network for predicting DNA methylation in multiple cells. DeepCpG has a modular architecture, consisting of a recurrent CpG module to account for correlations between CpG sites within and across cells, a convolutional DNA module to extract patterns from a wide DNA sequence window, and a Joint module that integrates the evidence from the CpG and DNA module to predict the methylation state of multiple cells for a target CpG site. DeepCpG yields accurate predictions, enables discovering DNA sequence motifs that are associated with DNA methylation states and cell-to-cell variability, and can be used for analyzing the effect of single-nucleotide mutations on DNA methylation. DeepCpG is implemented in Python and publicly available.

**Predicting DNA Methylation State of CpG Dinucleotide Using Genome Topological Features and Deep Networks** [[paper](http://www.nature.com/articles/srep19598)][[web server](http://dna.cs.usm.edu/deepmethyl/)]

This implementation uses a stacked autoencoder with a supervised layer on top of it to predict whether a certain type of genomic region called “CpG islands” (stretches with an overrepresentation of a sequence pattern where a C nucleotide is followed by a G) is methylated (a chemical modification to DNA that can modify its function, for instance methylation in the vicinity of a gene is often but not always related to the down-regulation or silencing of that gene.) This paper uses a network structure where the hidden layers in the autoencoder part have a much larger number of nodes than the input layer, so it would have been nice to read the authors’ thoughts on what the hidden layers represent.

### Single-cell applications <a name='genomics_single-cell'></a>

**DeepCpG - Predicting DNA methylation in single cells**
[[paper](http://dx.doi.org/10.1186/s13059-017-1189-z)]
[[code](https://github.com/cangermueller/deepcpg)]
[[docs](http://deepcpg.readthedocs.io/en/latest/)]

See above.

**CellCnn – Representation Learning for detection of disease-associated cell subsets**
[[code](https://github.com/eiriniar/CellCnn)][[paper](http://biorxiv.org/content/early/2016/03/31/046508)]

This is a convolutional network (Lasagne/Theano) based approach for “Representation Learning for detection of phenotype-associated cell subsets.” It is interesting because most neural network approaches for high-dimensional molecular measurements (such as those in the gene expression category above) have used autoencoders rather than convolutional nets.

**DeepCyTOF: Automated Cell Classification of Mass Cytometry Data by Deep Learning and Domain Adaptation**[[paper](http://biorxiv.org/content/biorxiv/early/2016/05/31/054411.full.pdf)]

Describes autoencoder approaches (stacked AE and multi-AE) to gating (assigning cells into discrete groups) with mass cytometry (CyTOF).

**Using Neural Networks To Improve Single-Cell RNA-Seq Data Analysis**[[preprint](http://biorxiv.org/content/early/2017/04/23/129759)]

Tests a variety of neural network architectures for obtaining a reduced representation of single-cell gene expression data. Introduces a database of tens of thousands of single-cell profiles which can be queried to infer a cell type or state based on this reduced representation.

**Removal of batch effects using distribution-matching residual networks**[[code](https://github.com/ushaham/BatchEffectRemoval)][[paper](https://academic.oup.com/bioinformatics/article-abstract/doi/10.1093/bioinformatics/btx196/3611270/Removal-of-Batch-Effects-using-Distribution)]

Most high-throughput assays in genomics, proteomics etc. are affected to some extent by systematic technical errors, so-called "batch effects". This paper uses a residual neural network to attenuate batch effects by trying to match the distributions of replicate experiments on e.g. single-cell RNA sequencing or mass cytometry. 

**Active deep learning reduces annotation burden in automatic cell segmentation** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/01/211060)]

Active learning, a framework addressing how to select training examples in order to train a model most efficiently, is shown to significantly reduce the time required by experts to annotate cell segmentation images in high-throughput high-context microscopy. Training deep learning models on this type of application of course requires a lot of high-quality labeled data, but the time of the human experts that can provide the labels (perform annotation) is limited and expensive. 

**scVAE: Variational auto-encoders for single-cell gene expression data** [[code](https://github.com/scvae/scvae)][[preprint](https://www.biorxiv.org/content/10.1101/318295v2)]

This approach models single-cell gene expression data directly from counts without initial normalization, and performs clustering in the latent space. Since it is based on a variational autoencoder, it can also be used to generate synthetic single-cell data by sampling from the latent distribution.

**Knowledge-primed neural networks enable biologically interpretable deep learning on single-cell sequencing data** [[code](https://github.com/epigen/KPNN)][[preprint](https://www.biorxiv.org/content/biorxiv/early/2019/10/07/794503.full.pdf)]

From the abstract: "Deep learning has emerged as a powerful methodology for predicting a variety of complex biological phenomena. However, its utility for biological discovery has so far been limited, given that generic deep neural networks provide little insight into the biological mechanisms that underlie a successful prediction. Here we demonstrate
deep learning on biological networks, where every node has a molecular equivalent (such as a protein or gene) and every edge has a mechanistic interpretation (e.g., a regulatory interaction along a signaling pathway). With knowledge-primed neural networks (KPNNs), we exploit the ability of deep learning algorithms to assign meaningful weights to multi-layered networks for interpretable deep learning."


### Population genetics <a name='genomics_pop'></a>

**Deep learning for population genetic inference** [[code](https://sourceforge.net/projects/evonet/)][[paper](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004845)]

**Diet networks: thin parameters for fat genomics** [[manuscript](http://openreview.net/pdf?id=Sk-oDY9ge)]

This weirdly-named paper addresses the frequently encountered problem in genomics where the number of features is much larger than the number of training examples. Here, it is addressed in the context of SNPs (single-nucleotide polymorphisms, genetic variations between individuals). The authors propose a new network parametrization that reduces the number of free parameters using a multi-task architecture which tries to learn a useful embedding of the input features.

### Systems biology<a name='sysbio'></a>

**Using deep learning to model the hierarchical structure and function of a cell** [[web server](http://d-cell.ucsd.edu)][[paper](https://www.nature.com/articles/nmeth.4627/)]

In this ambitious paper, the authors attempt to construct an interpretable neural network model (VNN; visible neural network) of a eukaryotic cell based on millions of genotype-phenotype associations. The network is built in a hierarchy with 12 levels, where each level is supposed to reflect a biologically meaningful level of organization. The resulting model can predict, for a given genetic perturbation, what the resulting phenotype is likely to be.

## Neuroscience <a name='neuro'></a>

There are potentially lots of implementations that could go here.

**Deep learning for neuroimaging: a validation study** [[paper](http://journal.frontiersin.org/article/10.3389/fnins.2014.00229/abstract)]

**SPINDLE: SPINtronic deep learning engine for large-scale neuromorphic computing** [[paper](http://dl.acm.org/citation.cfm?id=2627625)]

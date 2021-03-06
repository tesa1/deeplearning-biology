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

**(2018-11) A primer on deep learning in genomics** [[Nature Genetics paper](https://www.nature.com/articles/s41588-018-0295-5)]

This review, which features yours truly as one of its co-authors, is billed as a 'primer' which means it tries to help genomics researchers get started with deep learning. We tried to accomplish this by highlighting many practical issues such as tooling (not only deep learning libraries but also GPU cloud platforms, model zoos and online courses), defining your deep learning problem, explainability and troubleshooting. We also made a tutorial on Colaboratory that shows how to set up and run a simple convolutional network model for learning binding motifs, and how to inspect the model's predictions after it has been trained.



## Deeplearning used to better understand non-coding regions and (epi)genetic regulation

**(2015) DeepSEA – Predicting effects of noncoding variants with deep learning–based sequence model** [[paper](http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html)]

Like the packages above, this one also models chromatin accessibility as well as the binding of certain proteins (transcription factors) to DNA and the presence of so-called histone marks that are associated with changes in accessibility. This piece of software seems to focus a bit more explicitly than the others on predicting how single-nucleotide mutations affect the chromatin structure. Published in a high-profile journal (Nature Methods).

**(2015) DeepBind – Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning** [[paper](http://www.nature.com/nbt/journal/v33/n8/full/nbt.3300.html)]

This is from the group of Brendan Frey in Toronto, and the authors are also involved in the company Deep Genomics. DeepBind focuses on predicting the binding specificities of DNA-binding or RNA-binding proteins, based on experiments such as ChIP-seq, ChIP-chip, RIP-seq,  protein-binding microarrays, and HT-SELEX. Published in a high-profile journal (Nature Biotechnology.)

**(2016) DeepChrome: deep-learning for predicting gene expression from histone modifications** [[paper](https://academic.oup.com/bioinformatics/article/32/17/i639/2450757)]

Predicting gene expression from histone mark information (REMC database).

**(2016) DeeperBind - Enhancing Prediction of Sequence Specificities of DNA Binding Proteins** [[preprint](https://arxiv.org/pdf/1611.05777.pdf)]

This is an attempt to improve on DeepBind by adding a recurrent sequence learning module (LSTM) after the convolutional layer(s). In this way, the authors propose to capture a positional dimension that is lost in the pooling step in the original DeepBind design. They claim that benchmarking shows that this architecture leads to superior performance compared to previous work.

**(2016) DeepMotif - Visualizing Genomic Sequence Classifications** [[paper](https://arxiv.org/abs/1605.01133)]

This is also about learning and predicting binding specificities of proteins to certain DNA patterns or "motifs". However, this paper makes use of a combination of convolutional layers and [highway networks](https://arxiv.org/pdf/1505.00387v2.pdf), with more layers than the DeepBind network. The authors also show how a learned classifier can generate typical DNA motifs by input optimization; applying back-propagation with all the weights held constant in order to find an input pattern that maximally activates the appropriate output node in the network.

**(2016) Convolutional Neural Network Architectures for Predicting DNA-Protein Binding** [[paper](http://bioinformatics.oxfordjournals.org/content/32/12/i121.full)]

This work describes a systematic exploration of convolutional neural network (CNN) architectures for DNA-protein binding. It concludes that the convolutional kernels are very important for the success of the networks on motif-based tasks. Interestingly, the authors have provided a Dockerized implementation of DeepBind from the Frey lab (see above) and also provide EC2-laucher scripts and code for comparing different GPU enabled models programmed in Caffe.

**(2016) PEDLA: predicting enhancers with a deep learning-based algorithmic framework** [[paper](http://biorxiv.org/content/early/2016/01/07/036129)]

This package is for predicting enhancers (stretches of DNA that can enhance the expression of a gene under certain conditions or in a certain kind of cell, often working at a distance from the gene itself) based on heterogeneous data from (e.g.) the ENCODE project, using 1,114 features altogether.

**(2016) DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences** [[paper](https://academic.oup.com/nar/article/44/11/e107/2468300)]

Made for predicting the function of non-protein coding DNA sequence. Uses a convolution layer to capture regulatory motifs (i e single DNA snippets that control the expression of genes, for instance), and a recurrent layer (of the LSTM type) to try to discover a “grammar” for how these single motifs work together. Based on Keras/Theano.

**(2016) FIDDLE: An integrative deep learning framework for functional genomic data inference** [[preprint](http://biorxiv.org/content/early/2016/10/17/081380)]

The group predicted transcription start site and regulatory regions but claims this solution could be easily generalized and predict other features too. FIDDLE stands for Flexible Integration of Data with Deep LEarning. The idea is to model several genomic signals jointly using convolutional networks. This could be for example DNase-seq, ATAC-seq, ChIP-seq, TSS-seq, maybe RNA-seq signals (as in .wig files with one value per base in the genome).

**(2016) Basset – learning the regulatory code of the accessible genome with deep convolutional neural networks** [[paper](https://pubmed.ncbi.nlm.nih.gov/27197224/)]

Based on Torch, this package focuses on predicting the accessibility (or “openness”) of the chromatin – the physical packaging of the genetic information (DNA+associated proteins). This can exist in more condensed or relaxed states in different cell types, which is partly influenced by the DNA sequence (not completely, because then it would not differ from cell to cell.)

**(2017) Deep Learning Of The Regulatory Grammar Of Yeast 5′ Untranslated Regions From 500,000 Random Sequences** [[paper](https://genome.cshlp.org/content/27/12/2015.long)]

This is a CNN model that attempts to predict protein expression from the DNA sequence in a specific type of genomic region called 5' UTR (five-prime untranslated region). The model is built in Keras and a nice touch by the authors is that they optimized the parameters using hyperopt, which is also shown in one of the Jupyter notebooks that comes along with the paper. The results look promising and easily reproducible, judging from my own trial.

**(2017) Modeling Enhancer-Promoter Interactions with Attention-Based Neural Networks** [[preprint](https://www.biorxiv.org/content/early/2017/11/14/219667)]

The concept of attention in (recurrent) neural networks has become quite popular recently, not least because it has been used to great effect in machine translation models. This paper proposes an attention-based model for getting at the interactions between enhancer sequences and promoter sequences.

**(2018) Basenji – Sequential regulatory activity prediction across chromosomes with convolutional neural networks** [[paper](https://genome.cshlp.org/content/28/5/739.long)]

A follow-up project to Basset, this Tensorflow-based model uses both standard and dilated convolutions to model regulatory signals and gene expression (in the form of CAGE tag density) in many different cell types. 

**(2018) Genome-Wide Prediction of cis-Regulatory Regions Using Supervised Deep Learning Methods** (and several other papers applying various kinds of deep networks to regulatory region prediction) (one [[paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2187-1)] out of several)

Wyeth Wasserman’s group have made a kind of toolkit (based on the Theano tutorials) for applying different kinds of deep learning architectures to cis-regulatory element (DNA stretches that can modulate the expression of a nearby gene) prediction. They use a specific “feature selection layer” in their nets to restrict the number of features in the models. This is implemented as an additional sparse one-to-one linear layer between the input layer and the first hidden layer of a multi-layer perceptron.

**(2018) Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk** [[paper](https://www.nature.com/articles/s41588-018-0160-6)]

The so-called 'ExPecto paper'. The authors use a two-step model to predict the effect of genetic variants on gene expression. In the first step, the authors trained a convolutional neural network to model the 2002 epigenetic marks collected in ENCODE and ROADMAP consortium. In the second step, the authors trained a tissue-specific regularized linear model on the cis-regulatory region of the gene that is encoded by the first step convolutional neural network model. Then the effect of the variants on tissue-specific gene is calculated by the decrease in predicted gene expression through *in silico* mutagenesis.

**(2018) Predicting Transcription Factor Binding Sites with Convolutional Kernel Networks** [[paper](https://ieeexplore.ieee.org/document/8325519)]

This paper uses a hybrid of CNNs (to learn good representations) and kernel methods (to learn good prediction functions) to predict transcription factor binding sites.

**(2018) Predicting DNA accessibility in the pan-cancer tumor genome using RNA-seq, WGS, and deep learning** [[preprint](https://www.biorxiv.org/content/early/2017/12/05/229385)]

Like Basset (above) this paper shows how to predict DNA accessibility from sequence using CNNs, but it adds the possibility to leverage RNA sequencing data from different cell types as input. In this way implicit information related to cell type can be "transferred" to the accessibility prediction task.

**(2019) Deep learning at base-resolution reveals motif syntax of the cis-regulatory code** [[preprint](https://www.biorxiv.org/content/biorxiv/early/2019/08/21/737981.full.pdf)]

Here, a CNN with dilated convolutions is used to learn how different transcription factor binding motifs cooperate. This is the "motif syntax" mentioned in the title. The neural network is trained to predict the signal from a basepair-resolution ChIP assay (ChIP-nexus) and the trained network is then used to infer rules of motif cooperativity.

**(2019) DeepMILO: a deep learning approach to predict the impact of non-coding sequence variants on 3D chromatin structure** [[paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-01987-4)] 

The authors use both RNN and CNN models to predict the effects of variants on CTCF/cohesin-mediated insulator loops. They find mutations at loop anchors are associated with upregulation of the cancer driver genes BCL2 and MYC.

**(2020) Network-based machine learning in colorectal and bladder organoid models predicts anti-cancer drug efficacy in patients** [[paper](https://www.nature.com/articles/s41467-020-19313-8)]

Machine learning framework to identify robust drug biomarkers by taking advantage of protein-protein interaction networks using pharmacogenomic data derived from three-dimensional organoid culture models

**(2020) Supervised learning of gene-regulatory networks based on graph distance profiles of transcriptomics data** [[paper](https://www.nature.com/articles/s41540-020-0140-1)]

A supervised approach which utilises support vector machine to reconstruct gene regulatory networks (GRNs) based on distance profiles obtained from a graph representation of transcriptomics data. This may be applicable with Hi-C or GRO-seq or any sort of promoter/enhancer or contact/contact map.

**(2020) Predicting unrecognized enhancer-mediated genome topology by an ensemble machine learning model** [[preprint](https://www.biorxiv.org/content/10.1101/2020.04.10.036145v1.full)]

The authors constructed an ensemble machine learning model, LoopPredictor, to predict enhancer mediated loops in a genome-wide fashion across different cell lines and species.



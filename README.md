## ece792project
# Advanced Topics in ML Project

# Problem Statement:
Currently, machine learning (ML) image decompression (ID) models are trained with, and rely on, custom encoders to generate binary files for data storage. We believe that ML models can provide optimized lossy ID with the use of standard image compression (IC) techniques. We will use fractal encoding for IC and we will experiment with and benchmark Convolutional Neural Networks and Transformer Networks as the ID decoder model. We will evaluate our decoder model and compare it to the performance of state of the art (SOTA) ML ID stages. 

# Machine Learning Techniques (Experimental Plan): 
Advantage(s) over SOTA: SOTA models rely on CNN Auto-encoder architectures. Our team believes that additional performance gains are achievable in the lossy ID domain, namely decompression time and perceived image quality, with the use of CNN and Transformer Network models.
Initial Choice of Compression Algorithm: Fractal Coding. Our team will use the fractal encoding scheme as a starting point (https://github.com/pvigier/fractal-image-compression). Our project is not concerned with optimizing compression, therefore this method is chosen because it can produce binary data which is, by nature of the fractal encoding, proven decompressable through a series of iterative transformations.
Initial Choice for Decompression Machine Learning Model: CNN. Since fractal encoding breaks the image into blocks and fits transformations to the input data to arrive at a compressed representation, we believe that CNN’s will provide the best performance on the ML decompression task for this particular data.

# Performance Evaluation Benchmarks:
Lossy decompression is often measured as a trade off between decompression speed and image quality. Our team aims to experiment with a CNN ML model to benchmark its performance along the following metrics:
Image Quality Metrics including PSNR, MM-SSIM, etc.
Decoding Time - The LZ4 lossless ID algorithm can achieve around 400MB/s per core and Google’s Snappy can perform lossless ID at 500MB/s. JPEG image lossy decoding speeds range from 500MB/s to 1GB/s degrading quality.
Model Size

# General Plan of Work (Milestones)/ Division of Work: 
Project 1: Work on data collection, coding, testing, and analysis will be split evenly between team members. 
Collect image data that varies in scene, subject, and style.
Develop CNN ID model and test feasibility of ID task using the chosen encoding/decoding scheme.
Develop a benchmark testing suite for standardized testing of ID models, then evaluate our model.
Project 2: Continued work will be split evenly between team members.
Perform robustness testing on the model.
Perform an interpretability analysis of the model for the ID task.

# Proposed Citations: 
Yang, Yibo, Stephan Mandt, and Lucas Theis. "An introduction to neural data compression." arXiv preprint arXiv:2202.06533 (2022). 
DBLP Google Google Scholar MSAS Cite Key ZhouCGSW18 Statistics References: 0 Cited by: 0 Reviews: 0 Bibliographies: 0 PDF [Upload PDF for personal use] Researchr Researchr is a web site for finding, collecting, sharing, and reviewing scientific publications, for researchers by researchers.  Sign up for an account to create a profile with publication list, tag and review your related work, and share bibliographies with your co-authors.  Variational Autoencoder for Low Bit-rate Image Compression Lei Zhou, Chunlei Cai, Yue Gao, Sanbao Su, Junmin Wu. Variational Autoencoder for Low Bit-rate Image Compression. In 2018 IEEE Conference on Computer Vision and Pattern Recognition Workshops, CVPR Workshops 2018, Salt Lake City, UT, USA, June 18-22, 2018. pages 2617-2620, IEEE Computer Society, 2018.
Wang, Dezhao, et al. "Neural Data-Dependent Transform for Learned Image Compression." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
Cheng, Zhengxue, et al. "Learned lossless image compression with a hyperprior and discretized gaussian mixture likelihoods." ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020.


# Road Map

### Phase 1 (due 2/27)

- [x] Compressing Images
    - [x] Understand the fractal compression code
        - [ ] Create overview of code, so others in group can understand
    - [x] Implement the compression algo to begin building a compressed data set to use for large scale traning of model
        - Is Github an appropiate place to store these images? 

- [ ] Buiding Benchmark Suite
    - [ ] Research analysis benchmarks for:
        - image decoding speed
        - image quaity (mulitple and varying)
        - Note: Use papers so we can have an "apples to apples" comparison
    - [ ] Plan architecture for benchmark suite
    - [ ] Build code to test images
        - Focus on later integration into model (we could possible use these functions as loss functions)

- [x] Building "Decoding Model"
    - [x] Research futher current methods
    - [x] Get basic basline working to ensure we can decompress a compressed image
    - [x] Increase complexity of the model to increase scores based on benchmark (this relies on creaitng benchmark loss)


### Phase 2
Main goal is to create benchmark suite and increase the number of models to try. 
- Continuously determine models to try
- Talk about what our next plans are as a group



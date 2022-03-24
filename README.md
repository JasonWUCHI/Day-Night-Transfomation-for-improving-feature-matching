## COMP4471-Deep-Learning-Computer-Vision

## pa1: KNN, SVM, Softmax, Neural Network
  Raw Numpy implementation of ML algorithms + Numpy Vectorization for performance optimization </br>
  Dataset: run cs231n/datasets/get_datasets.py

## pa2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets
  Raw Numpy implementation of Batch Normalization, Dropout, CNN + Numpy Vectorization for performance optimization </br>
  Dataset: run cs231n/datasets/get_datasets.py
  
## pa3: Vanilla RNN, LSTM, RNN on Language Models, Image Captioning, GAN, Style Transfer, Network Visualisation
  Raw Numpy implementation of RNN and apply them for various applications </br>
  Dataset: run cs231n/datasets/get_assignment3_data.py

## Final Project: Day-night Image Transformation Using CycleGAN and D2-net
### Abstract 
  Illumination variability is arguably one of the most crucial factors in image matching, a task in which aligning structure, pattern, and content between photos remain a hurdle to overcome. A sharp difference in illumination between images can significantly compromise the matching performance, in particular the matching of the images taken at day and those at night. In this study, we propose a GAN-based, image-to-image translation network to tackle the day-night feature matching challenge. Our modified CycleGAN model transforms day images to night ones and vice versa, followed by analysis using D2-net descriptors. By comparing our modified CycleGAN model to the vanilla CycleGAN and the OpenCV models, the proposed model produces greater image quality and delivers better performance on feature matching.

### Proposed Algorithm

An intuitive algorithm that performs day-to-night (or night-to-day) transformation is proposed in this study, in which the transformed images and the other night/day photos will serve as the input for the downstream matching task. For the image transformation stage, a CycleGAN model is used for training (with some detailed modifications, please refer to the report pdf) since it is known for learning style transferring between images. Its strength is exemplified by mapping horses to zebras, Van Gogh’s painting to Monet’s, and, in our case, day images to night images. Our CycleGAN model will be trained on the day images and night images from the 24/7 Tokyo dataset \[10\]. For the feature matching algorithm, the pre-trained D2-net model is chosen, given its excellent performance in day-night localization tasks. The Aachen Day-Night dataset \[11\] is used for our final evaluation, where a night query image is paired with a transformed-day image and a transformednight query is paired with a day image. In the end, the results from the two pipelines will be aligned, and those with a matching score above a given threshold will be retained. </br>

<p align="center">
  
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/cyclegan_transform.jpeg"  width="500">
</br>
CycleGAN 
</br>
</br>

<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/D2net.png"  width="500"> 
</br>
D2Net
</br>
</br>

<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/pipeline.png"  width="300"> 
</br>
Algorithm Pipeline
</br>

</p>
</br>

### GAN Training Results (Day -> Night)

\[Training 400 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-000400-Y-X.png"  width="300"> 

\[Training 2000 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-002000-Y-X.png"  width="300"> 

\[Training 3600 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-003600-Y-X.png"  width="300"> 

\[Training 6400 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-006400-Y-X.png"  width="300"> 

\[Training 13600 iterations...\] </br>
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/sample-013600-Y-X.png"  width="300"> 
</br>

### GAN Results vs OpenCV brightness 

This section shows the generated image comparison from GAN compared with brightness modification using OpenCV.

</br>
<p align="center">
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/result_compare.png"  width="600"> 
</br>
Generated image comparison
</br>
  
</p>
</br>

### Feature Matching Results

This section shows the feature matching results from GAN compared with brightness modification using OpenCV.

</br>
<p align="center">
<img src="https://github.com/PeePeeDante/COMP4471-Deep-Learning-Computer-Vision/blob/main/pictures/matching_compare.png"  width="600"> 
</br>
Feature matching comparison
</br>
  
</p>
</br>

### Reference
</br>
[1] Zhenfeng Shao, Min Chen, and Chong Liu. Feature matching for illumination variation images. Journal of Electronic Imaging, 24:033011, 05 2015. </br>
[2] Hao Zhou, Torsten Sattler, and David Jacobs. Evaluating Local Features for Day-Night Matching, pages 724–736. 11 2016. </br>
[3] Mihai Dusmanu, Ignacio Rocco, Tomas Pajdla, Marc Pollefeys, Josef Sivic, Akihiko Torii, and Torsten Sattler. D2-net: A trainable cnn for joint detection and description of local features, 2019. </br>
[4] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks, 2020. </br>
[5] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks, 2018. </br>
[6] Taeksoo Kim, Moonsu Cha, Hyunsoo Kim, Jung Kwon Lee, and Jiwon Kim. Learning to discover cross-domain relations with generative adversarial networks, 2017.</br> 
[7] Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, and Jaegul Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation, 2018. </br>
[8] Yangyun Shen, Runnan Huang, and Wenkai Huang. Gdstargan: Multi-domain image-to-image translation in garment design. PLOS ONE, 15(4):1–15, 04 2020. </br>
[9] Sefik Eskimez, Dimitrios Dimitriadis, Kenichi Kumatani, and Robert Gmyr. One-shot voice conversion with speakeragnostic stargan. pages 1334–1338, 08 2021. </br>
[10] A. Torii, R. Arandjelovic, J. Sivic, M. Okutomi, and T. Pajdla. 24/7 place recognition by view synthesis. In CVPR, 2015. </br>
[11] Aachen day-night dataset, 2021. </br>
[12] Judy Hoffman, Eric Tzeng, Taesung Park, Jun-Yan Zhu, Phillip Isola, Kate Saenko, Alexei A. Efros, and Trevor Darrell. Cycada: Cycle-consistent adversarial domain adaptation, 2017. </br>

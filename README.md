# Source Acquisition Device Identification from Recorded Audio Based on Spatio-temporal Representation Learning with Multi-Attention Mechanisms

---

## Table of Contents
---
1. [Project description](#description)
2. [Dependencies](#dependencies)
3. [Our process](#process)
4. [Acknowledgements](#acknowledgements)
<br>


## <h2 id="description"> Project description:</h2>
---
This repository contains the code release for our paper titled as "Source Acquisition Device Identification from Recorded Audio Based on Spatio-temporal Representation Learning with Multi-Attention Mechanisms".  
<br>
The code was developed in Matlab and Python. The code for the extraction of the input features MFCC and GSV is used on Matlab , and the training and testing of the model is used on Python. The code is aimed to provide the implementation of the framework proposed in our paper to perform the identification of the audio acquisition source devices.  
<br>
Among the functions used for feature extraction, the melcepst function is from Voicebox Speech Signal Processing Toolkit, the gmm_em function and the mapAdapt function are from MSR Identity Toolkit.  
<br>


## <h2 id="dependencies"> Dependencies:</h2>
---
Python==3.6  
TensorFlow==2.1.0   
Keras==2.3.1  
Numpy==1.17.0  
Pandas==1.1.3  
<br>

## <h2 id="process"> Our process:</h2>
---
- We use a dataset consisting of audio samples recorded by different devices, with the recording acquisition device as the label.
- We extract two kinds of features by get_MFCC and get_GSV, and the parameter settings are shown in the paper.
- We train a spatial-temporal representation learning model with a multi-attention mechanism to predict the correct recording acquisition device for a given audio sample.


## <h2 id="acknowledgements"> Acknowledgements:</h2>
---
<br>
http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html<br>
https://www.microsoft.com/en-us/download/details.aspx?id=52284<br>
If you have any question, please feel free to contact us through zfwang@ccnu.edu.cn





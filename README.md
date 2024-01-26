______________________________________________________________________

<div align="center">

# RTFS-Net: Recurrent time-frequency modelling for efficient audio-visual speech separation

[![arXiv](https://img.shields.io/badge/arXiv-2306.00160-brightgreen.svg)](https://arxiv.org/abs/2309.17189)
[![Samples](https://img.shields.io/badge/Website-Demo_Samples-blue.svg)](https://anonymous.4open.science/w/RTFS-Net/AV-Model-Demo.html)

</div>

Welcome to the official GitHub repository for RTFS-Net: Recurrent time-frequency modelling for efficient audio-visual speech separation.


## Abstract

Audio-visual speech separation methods aim to integrate different modalities to generate high-quality separated speech, thereby enhancing the performance of downstream tasks such as speech recognition. Most existing state-of-the-art (SOTA) models operate in the time domain. However, their overly simplistic approach to modeling acoustic features often necessitates larger and more computationally intensive models in order to achieve SOTA performance. In this paper, we present a novel time-frequency domain audio-visual speech separation method: Recurrent Time-Frequency Separation Network (RTFS-Net), which applies its algorithms on the complex time-frequency bins yielded by the Short-Time Fourier Transform. We model and capture the time and frequency dimensions of the audio independently using a multi-layered RNN along each dimension. Furthermore, we introduce a unique attention-based fusion technique for the efficient integration of audio and visual information, and a new mask separation approach that takes advantage of the intrinsic spectral nature of the acoustic features for a clearer separation. RTFS-Net outperforms the previous SOTA method using only 10\% of the parameters and 18\% of the MACs. This is the first time-frequency domain audio-visual speech separation method to outperform all contemporary time-domain counterparts.

**Problem:** The 'cocktail party problem' highlights the difficulty machines face in isolating a single voice from overlapping conversations and background noise, a task easily managed by humans. Existing Audio-only Speech Separation (AOSS) methods struggle particularly in noisy environments with heavy voice overlap.

**Issues with Current Methods:** 
- **T-domain Methods:** These offer high-quality audio separation but suffer from high computational complexity and slow processing due to their extensive parameter count.
- **TF-domain Methods:** More efficient in computation but historically underperform compared to T-domain methods. They face three key challenges:
  1. Lack of independent modeling of time and frequency dimensions.
  2. Insufficient use of visual cues from multiple receptive fields for enhancing model performance.
  3. Poor handling of complex features, leading to loss of critical amplitude and phase information.

**Proposed Solution: Recursive Time-Frequency Separation Network (RTFS-Net)**
- **Approach:** Integrates audio and high-fidelity visual cues using a novel TF-domain method.
- **Innovations:**
  1. **RTFS Blocks:** Compress and independently model acoustic dimensions (time and frequency), minimizing information loss while creating a low-complexity subspace.
  2. **Cross-dimensional Attention Fusion (CAF) Block:** Efficiently fuses audio and visual information for enhanced voice separation while utilizing 1.3% the computational complexity of the previous SOTA method.
  3. **Spectral Source Separation ($S^3$) Block:** Effectively extracts the target speaker's voice features using complex numbers.
 
## Results and Comparison

Comparison of RTFS-Net with Existing AVSS Methods.

### Table for LRS2-2Mix Dataset
|     Model     | SI-SNRi | SDRi | PESQ | Params |  MACs |  Time |
|:-------------:|:-------:|:----:|:----:|:------:|:-----:|:-----:|
|   CaffNet-C*  |    -    | 12.5 | 1.15 |    -   |   -   |   -   |
|   Thanh-Dat   |    -    | 11.6 |   -  |    -   |   -   |   -   |
| AV-ConvTasnet |   12.5  | 12.8 |   -  |  16.5  |   -   |  60.3 |
|  VisualVoice  |   11.5  | 11.8 | 2.78 |  77.8  |   -   | 130.2 |
|     AVLIT     |   12.8  | 13.1 | 2.56 |   5.8  |  36.4 |  53.4 |
|     CTCNet    |   14.3  | 14.6 | 3.08 |   7.0  | 167.2 | 122.7 |
|   RTFS-Net-4  |   14.1  | 14.3 | 3.02 |   0.7  |  21.9 |  57.8 |
|   RTFS-Net-6  |   14.6  | 14.8 | 3.03 |   0.7  |  30.5 |  64.7 |
|  RTFS-Net-12  |   14.9  | 15.1 | 3.07 |   0.7  |  56.4 | 109.9 |

### Table for LRS3-2Mix Dataset
|     Model     | SI-SNRi | SDRi | PESQ | Params |  MACs |  Time |
|:-------------:|:-------:|:----:|:----:|:------:|:-----:|:-----:|
|   CaffNet-C*  |    -    | 12.3 |   -  |    -   |   -   |   -   |
|   Thanh-Dat   |    -    |   -  |   -  |    -   |   -   |   -   |
| AV-ConvTasnet |   11.2  | 11.7 |   -  |  16.5  |   -   |  60.3 |
|  VisualVoice  |    9.9  | 10.3 |   -  |  77.8  |   -   | 130.2 |
|     AVLIT     |   13.5  | 13.6 | 2.78 |   5.8  |  36.4 |  53.4 |
|     CTCNet    |   17.4  | 17.5 | 3.24 |   7.0  | 167.2 | 122.7 |
|   RTFS-Net-4  |   15.5  | 15.6 | 3.08 |   0.7  |  21.9 |  57.8 |
|   RTFS-Net-6  |   16.9  | 17.1 | 3.12 |   0.7  |  30.5 |  64.7 |
|  RTFS-Net-12  |   17.5  | 17.6 | 3.25 |   0.7  |  56.4 | 109.9 |

### Table for VoxCeleb2-2Mix Dataset
|     Model     | SI-SNRi | SDRi | PESQ | Params |  MACs |  Time |
|:-------------:|:-------:|:----:|:----:|:------:|:-----:|:-----:|
|   CaffNet-C*  |    -    |   -  |   -  |    -   |   -   |   -   |
|   Thanh-Dat   |    -    |   -  |   -  |    -   |   -   |   -   |
| AV-ConvTasnet |    9.2  |  9.8 |   -  |  16.5  |   -   |  60.3 |
|  VisualVoice  |    9.3  | 10.2 |   -  |  77.8  |   -   | 130.2 |
|     AVLIT     |    9.4  |  9.9 | 2.23 |   5.8  |  36.4 |  53.4 |
|     CTCNet    |   11.9  | 13.1 | 3.00 |   7.0  | 167.2 | 122.7 |
|   RTFS-Net-4  |   11.5  | 12.4 | 2.94 |   0.7  |  21.9 |  57.8 |
|   RTFS-Net-6  |   11.8  | 12.8 | 2.97 |   0.7  |  30.5 |  64.7 |
|  RTFS-Net-12  |   12.4  | 13.6 | 3.00 |   0.7  |  56.4 | 109.9 |

Note: Larger SI-SNRi and SDRi values indicate better performance. - indicates results not reported in the original paper. * indicates audio reconstructed using the ground-truth phase.

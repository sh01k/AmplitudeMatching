# Amplitude Matching

## Description
Amplitude matching is a multizone sound field control method to synthesize desired amplitude (or magnitude) distributions over a target region with multiple loudspeakers. This repository provides codes for reproducing results in the following article. 

- T. Abe, S. Koyama, N. Ueno, and H. Saruwatari, "Amplitude Matching for Multizone Sound Field Control," *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, DOI: 10.1109/TASLP.2022.3231715, 2020.

The article is open access on [IEEE Xplore](https://doi.org/10.1109/TASLP.2022.3231715).

#### Abstract
A multizone sound field control method, called amplitude matching, is proposed. The objective of amplitude matching is to synthesize a desired amplitude (or magnitude) distribution over a target region with multiple loudspeakers, whereas the phase distribution is arbitrary. Most of the current multizone sound field control methods are intended to synthesize a specific sound field including phase or to control acoustic potential energy inside the target region. In amplitude matching, a specific desired amplitude distribution can be set, ignoring sound propagation directions. Although the optimization problem of amplitude matching does not have a closed-form solution, our proposed algorithm based on the alternating direction method of multipliers (ADMM) allows us to accurately and efficiently synthesize the desired amplitude distribution. We also introduce the differential-norm penalty for a time-domain filter design with a small filter length. The experimental results indicated that the proposed method outperforms current multizone sound field control methods in terms of accuracy of the synthesized amplitude distribution. 

## Usage
Each jupyter notebook file corresponds to the section of the article. To run the codes for Section VI-C ([Sect-VI-C.ipynb](https://github.com/sh01k/AmplitudeMatching/blob/main/Sect-VI-C.ipynb)), [MeshRIR dataset](https://www.sh01.org/MeshRIR/) is required. Place irutilities.py and S32-M441_npy in the same directory.

## License
[MIT](https://github.com/sh01k/AmplitudeMatching/blob/main/LICENSE)

## Author
- Takumi Abe
- [Shoichi Koyama](https://www.sh01.org) 
- [Natsuki Ueno](https://natsuenono.github.io/)
- [Hiroshi Saruwatari](https://researchmap.jp/read0102891/)
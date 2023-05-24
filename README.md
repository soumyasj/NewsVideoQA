# NewsVideoQA

This repository provides code implementations for baselines and links to the proposed dataset mentioned in the paper "Watching the News: Towards VideoQA Models that can Read" (WACV-2023).

[![arXiv: 2211.05588](https://img.shields.io/badge/arXiv-2211.05588-brightgreen.svg)](https://arxiv.org/abs/2211.05588) [![webpage: CVIT](https://img.shields.io/badge/webpage-CVIT-blue.svg)](http://cvit.iiit.ac.in/research/projects/cvit-projects/videoqa) [![video: YouTube](https://img.shields.io/badge/video-YouTube-red.svg)](https://www.youtube.com/watch?v=rnCCONldMik)  [![Dataset: RRC](https://img.shields.io/badge/dataset-RRC-orange.svg)](https://rrc.cvc.uab.es/?ch=24&com=downloads)


## Task

Video Question Answering methods focus on commonsense reasoning and visual cognition of objects or persons and their interactions over time. Current VideoQA approaches ignore the textual information present in the video. We introduce the ``NewsVideoQA'' dataset that comprises more than 8,600+ QA pairs on 3,000+ news videos obtained from diverse news channels from around the world.

<p align="center">
  <img src="https://github.com/soumyasj/NewsVideoQA/blob/main/images/task.png?raw=true" alt="Task" width="600">
</p>

## Samples from the Dataset
<p align="center">
  <img src="https://github.com/soumyasj/NewsVideoQA/blob/main/images/few_examples_from_dataset.png?raw=true" alt="Task" width="600">
</p>

## Baselines

1. BERT: `baselines/BERT`
2. M4C: `baselines/M4C`
3. SINGULARITY: `baselines/SINGULARITY`

# Citation
If you find our dataset/code useful, feel free to leave a star and please cite our paper as follows:
```
@inproceedings{DBLP:conf/wacv/JahagirdarMKJ23,
  author       = {Soumya Jahagirdar and
                  Minesh Mathew and
                  Dimosthenis Karatzas and
                  C. V. Jawahar},
  title        = {Watching the News: Towards VideoQA Models that can Read},
  booktitle    = {{IEEE/CVF} Winter Conference on Applications of Computer Vision, {WACV}
                  2023, Waikoloa, HI, USA, January 2-7, 2023},
  pages        = {4430--4439},
  publisher    = {{IEEE}},
  year         = {2023},
}
```

## Contact
For any clarifications, comments, or suggestions, please create an issue or contact [Soumya Shamarao Jahagirdar](https://www.linkedin.com/in/soumya-jahagirdar/).




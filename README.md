# Unsupervised Decomposition of Multi-Author Documents

This repository contains the research paper and code for an NLP project on **unsupervised author segmentation in multi-author documents**, conducted at the Indian Institute of Technology Delhi.

<p align="center">
<img src="Images/Readme.png">
</p>

## Paper

**Title:** Unsupervised Decomposition of Multi-Author Document: Exploiting the Difference of Syntactic Writing Styles

**Authors:** Kautsya Kanu, Sayantan Sengupta

**Context:** Research conducted under the guidance of [Prof. Mausam](https://www.cse.iitd.ac.in/~mausam/) as part of the COL 772 (Natural Language Processing) graduate course at IIT Delhi, 2016. Made available as a preprint in 2019.

📄 **Read the paper:** [paper.pdf](./paper.pdf)

## Problem

Given a document collaboratively written by multiple authors, decompose it into authorial components — *without* prior knowledge of the authors or their writing samples. The motivating application is plagiarism detection: detecting when a submitted document is composed of segments written by different authors, even when each author writes in a different style about the same topic.

This is a hard unsupervised problem because:

- Topic-based decomposition is well-studied, but topic alone doesn't separate multiple authors writing on the same subject.
- Without ground-truth author samples, supervised classification approaches don't apply.
- The number of authors is typically unknown.

## Approach

The work proposes two contributions over the Akiva & Koppel (2013) baseline, which used a bag-of-words feature model:

### 1. Syntactic style features via PQ-grams

The bag-of-words model fails to distinguish syntactically distinct sentences with similar lexical content. For example:

> S1: "My chair started squeaking a few days ago, and it's driving me nuts."
>
> S2: "Since a few days ago, my chair has been squeaking — it's simply annoying."

These convey the same meaning with very different grammatical structures. To capture this, the paper uses:

- **PQ-grams** extracted from sentence parse trees (via Stanford Parser) to represent syntactic substructures
- **POS tag n-grams** (bigrams and trigrams) on Penn Treebank tags, deliberately excluding head words to capture structure rather than vocabulary choice

Each segment is represented as a frequency profile over its PQ-gram index, providing a stable, vocabulary-independent signature of each author's syntactic style.

### 2. Similarity-index filtering for segment purity

The baseline pipeline feeds *all* segments into the GMM clustering step, but mixed segments (containing sentences from multiple authors) hurt clustering precision. This work introduces a **similarity index** computed by counting shared PQ-grams between sentence pairs within a segment, normalized by their total PQ-gram counts.

Segments with low similarity scores are likely mixed and are excluded from the initial GMM training. Pure segments train the GMM; mixed segments are then classified using posterior probabilities. Vital segments from each cluster train a final Naive Bayes classifier.

## Results

Evaluation on two datasets:

- **Becker-Posner blog corpus:** 690 blogs by Gary Becker and Richard Posner (26,922 sentences, 2 authors)
- **NYT corpus:** 1,182 articles by Maureen Dowd, Gail Collins, Thomas Friedman, and Paul Krugman (varying author counts)

Key finding: the proposed PQ-gram and POS-tag features show substantially better **robustness** to chunk-size variation than the baseline. The baseline's accuracy on Becker-Posner drops from 0.82 (V=200) to 0.51 (V=50), while POS-tag trigrams hold at 0.70 / 0.67 / 0.65 across the same range. The features trade peak accuracy for stability — a meaningful property when the chunk size in real plagiarism detection is unknown a priori.

| Features | V=200 | V=100 | V=50 |
|----------|-------|-------|------|
| Baseline (bag-of-words) | 0.82 | 0.57 | 0.51 |
| POS-tags (bigram) | 0.67 | 0.64 | 0.63 |
| POS-tags (trigram) | 0.70 | 0.67 | 0.65 |
| PQ-Grams | 0.66 | 0.63 | 0.62 |
| PQ-Grams + TF-IDF | 0.69 | 0.65 | 0.63 |

## Author Contributions

My primary contributions to this work:

- Introduced PQ-gram-based features for capturing authorial syntactic style
- Implemented and evaluated the hypothesis that training the GMM on filtered "pure" segments (rather than all segments) improves classifier reliability
- Led the writing of the paper, under the guidance of co-author Sayantan Sengupta (a prospective Ph.D. student under Prof. Mausam at the time)

## References

- Akiva, N., & Koppel, M. (2013). A generic unsupervised method for decomposing multi-author documents.
- Aldebei, K., He, X., & Yang, J. Unsupervised Decomposition of a Multi-Author Document Based on Naive-Bayesian Model. *ACL*.
- Tschuggnall, M., & Specht, G. Automatic Decomposition of Multi-Author Documents Using Grammar Analysis. University of Innsbruck.

See the [paper](./paper.pdf) for the full reference list.

## Requirements

  * python (>2.7.11)
  * scikit-learn
  * nltk
  * numpy

**Note:** This project was completed in 2016 using Python 2.7 and the dependencies of that era. The code is preserved in its original form

## Usage
#### Baseline
python Src\ Code/baseline.py<br />
#### Improvements
python Src\ Code/Our\ Methods/words_method.py<br />
python Src\ Code/Our\ Methods/parser_method.py<br />
python Src\ Code/Our\ Methods/hybrid_words_parser_method.py<br />

## Citation

If you reference this work, please cite as:

```
Kanu, K., & Sengupta, S. (2016). Unsupervised Decomposition of Multi-Author Document:
Exploiting the Difference of Syntactic Writing Styles. Research conducted at
IIT Delhi under the supervision of Prof. Mausam.
```

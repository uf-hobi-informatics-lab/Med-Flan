This README describes the data for the n2c2 Track 2 challenge, Extracting Social Determinants of Health (SDOH). This release only includes the training data associated with Subtask A, which is focused on in-domain information extraction (see https://n2c2.dbmi.hms.harvard.edu/2022-track-2). Below is a description of the directory contents:

################################################################################
Manuscript
################################################################################
The file, "SHAC Manuscript.pdf," is the original publication describing the Social History Annotated Corpus (SHAC) used in this challenge. The manuscript describes the annotation scheme, data sources and sampling, inter-rater agreement, and initial extraction results. As part of the preparation of SHAC for this challenge, the annotations were carefully reviewed and adjusted, to improve the quality of the annotations. The reported inter-rater agreement in the manuscript does not reflect these additional quality checks, which were performed after the paper publication. This manuscript can be cited as:

Kevin Lybarger, Mari Ostendorf, and Meliha Yetisgen, "Annotating social determinants of health using active learning, and characterizing determinants using neural event extraction," Journal Biomedical Informatics, vol. 113, p. 103631, 2021. DOI: 10.1016/j.jbi.2020.103631

################################################################################
Annotation guidelines
################################################################################
The file, "SHAC Guidelines 20220207.pdf," is a truncated version of the original annotation guidelines. The original SHAC corpus was annotated for 12 SDOH. This challenge focuses on the five most prevalent SDOH, including alcohol use, drug use, and tobacco use, employment, and living situation. These guidelines only include the guidance associated with these five SDOH.

################################################################################
Annotations
################################################################################
The data was annotated and is provided in BRAT format (see https://brat.nlplab.org/). Each annotated document consists of a text file (*.txt) and annotation files (*.ann). During evaluation, model predictions shall be provided in the same format.

The annotations are in the, "Annotations," directory. This directory includes the BRAT annotation configuration file ("annotation.conf"), visualization configuration file ("visual.conf"), and tools configuration file ("tools.conf").

The SHAC training set was 29% random sampled and 71% actively sampled (see manuscript for details). The development and test sets were exclusively randomly sampled. The MIMIC training set (1,316 documents) and development set (188 documents) are provided at this stage. The training set includes actively selected samples, which tend to describe more complex SDOH. The development set is randomly selected, similar to the test set.

################################################################################
Evaluation criteria and scoring
################################################################################
The evaluation criteria and a Python scoring script will be posted to the challenge website.

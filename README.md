# Statement Verification For Tables

This repository contains models and datasets written as a part of the SEMEVAL 2021 Task9 - **Statement Verification &
Evidence Finding with Tables**.

Please visit [paper website](https://atahanak.github.io/statement_verification_for_tables/) for the complete paper.

# Abstract

Information dirt is a severe problem that needs to be addressed. However, with the rise of social media, manually monitoring the validity of information has become infeasible. Thus fact verification is heavily investigated in the literature. However, structured (table) data-based fact verification remains mostly unexplored. To the best of our knowledge, the first dataset for table based fact verification is introduced recently by  [@chen2020tabfact]. As with most downstream natural language processing tasks, fact verification is also improved by large-scale pre-trained language models. [@chen2020tabfact] showed how BERT can be leveraged for table based fact verification. Moreover, [@tapasf, @SAT] explored different ways to inject structural information into BERT, which substantially improved the performance of the model. In this work, we judiciously analyze various sequencing strategies and find that related information must be close to each other. Furthermore, we propose a two-level approach for table-based fact verification, which paves the way for specialized models for deciding whether the evidence contains sufficient information and whether the claim is entailed or refuted. It also enables to generate more training data without disturbing the balance and the quality of the dataset.



# Hierarchical Multi-class and Multi-label Text Classification for Crime Report: A Deep Learning Ensemble Analysis
This article addresses the hierarchical multi-class and multi-label classification of criminal narratives.
We propose an ensemble-based framework that integrates traditional machine learning techniques with large language models (LLMs).
The approach employs XGBoost classifiers, a pre-trained Llama 3.1 8B model, and a fine-tuned Llama variant.
Each model performs predictions across three hierarchical levels.
We compute probabilities for all decisions and consolidate the results into a directed acyclic graph (DAG).
The ensemble selects the final prediction based on the highest aggregated confidence, and multiple combinatorial scenarios involving the analyzed models.
Experiments are conducted on a large dataset of Rio de Janeiro State Civil Police.
Results demonstrate that the fine-tuned Llama model outperforms the pre-trained version, improving accuracy by 36\% and reducing convergence failures.
XGBoost maintains stable performance and complements LLMs in ensemble scenarios.
The combination of models enhances coverage and recovers cases that would otherwise be discarded due to individual model failures.
Statistical tests confirm significant gains from fine-tuning and indicate that non-tuned models may degrade ensemble performance.
When fine-tuning is not feasible, we recommend combining traditional machine learning with language models in an ensemble configuration.
The proposed methodology improves robustness, stability, and reliability in operational applications.
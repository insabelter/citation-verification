# Automated Citation Verification in Scientific Literature Using Large Language Models

This repository contains the source code for my master's thesis on the influence of linguistic features of citation statements on automated citation verification when using a zero-shot in-context learning LLM classification approach. 

The dataset and implemented architecture build upon the research work of Tianmai M. Zhang and Neil F. Abernethy: *"Detecting Reference Errors in Scientific Literature with Large Language Models"* (https://arxiv.org/abs/2411.06101)

## Instructions
The master's thesis work should be replicatable using the provided Jupyter Notebooks and methodology instructions from the master's thesis.

The dataset, paper PDFs and dataframes with LLM prompt responses are included in the repository to be able to directly display the achieved results.

In order to perform the reference paper chunk retrieval or the OpenAI model prompting, an OpenAI API key needs to be setup (see https://platform.openai.com/api-keys) and pasted into the file 'open_ai_key.txt'.

The models can be prompted via the automated prompting python scripts.
In order to not cancel the process on a remote session logout, it can be useful to run them using a nohup command (from within the scripts/ folder):
`nohup python prompt_openai_model.py > nohup_logs/<log_name>.log 2>&1 &`

### Dependencies:
- **Standard Library:**
glob, json, os, re, sys, xml.etree.ElementTree
- **Data Processing & Analysis:**
pandas, numpy, scipy
- **Web & Requests:**
lxml, requests
- **Visualization:**
matplotlib, IPython.display
- **LLM-related:**
llama_index, ollama, openai

## Repository Structure
### Notebooks:
1. **Dataset Analysis:**
    Plots of data distributions within the reannotated dataset
2. **PDF Text Extraction:**
    PDF to TEI conversion using GROBID + raw paper body text extraction (PBTE)
3. **Reference Paper Chunk Retrieval:**
    Retrieving top three most relevant chunks for each reference paper using IndexLlama
4. **Prompt Creation:**
    Testing the prompt creation functions from 'create_prompts.py' script on examples
5. **OpenAI Model Prompting:**
    Notebook for prompting an OpenAI model step-wise and manually checking its progress (alternative to automated 'prompt_openai_model.py' script)
6. **Model Responses Evaluation:**
    Gathering results from all parameter configurations for all models and saving it to 'all_results.json'
7. **Results Evaluation and Visualization:**
    Notebook containing results for configuration parameter analyses and significance testing of annotation attribute value group performance differences

### Scripts:
- **create_prompts.py:**
    Helper functions to create the classification prompt based on a dataset row
- **data_visualizations.py:**
    Helper functions for visualizations for dataset and results analyses
- **df_calculations.py:**
    Helper functions for loading dataset dataframes (with llm answers) and evaluating the answers by calculating the performance metrics
- **prompt_large_local_model.py:**
    Script for automated prompting of local ollama model (suited for nohup command)
- **prompt_openai_model.py:**
    Script for automated batched prompting of OpenAI model (suited for nohup command)
- **significance_tests.py:**
    Helper functions for performing the significance tests for the results evaluation

## Data:
- **ReferenceErrorDetection_data.xlsx:**
    Original Dataset by Zhang et al. (downloaded from https://github.com/tianmai-zhang/ReferenceErrorDetection)
- **ReferenceErrorDetection_data_extended_annotation.xlsx:**
    Cleaned and reannotated dataset used for experiments of master's thesis
- **all_results.json:**
    JSON file containing evaluated metrics for all tested parameter configurations
- **results_per_attribute.json:**
    JSON file containing metrics per annotation attribute for best classification configuration (LLM: GPT-4.1)
- **papers/:**
    All Downloaded PDFs for reference and citation papers
- **extractions/:**
    Extracted reference paper texts in TEI format and raw body text converted to .txt files
- **batch_responses/:**
    LLM responses for all configurations of classification prompting
- **dfs/:**
    Pickled dataset dataframes extended with model responses for results evaluation

## Master's Thesis Abstract
Despite the importance of accurate citations for scientific literature quality, citation errors
persist across scholarly publications. This thesis investigates the potential of zero-shot in-context learning with large language models (LLMs) for automated citation verification, specifically examining how linguistic citation statement characteristics influence classification performance. An existing dataset of 250 citation-reference pairs was re-annotated to extract syntactic and semantic citation features. Building on prior work, a classification pipeline was implemented with systematic optimizations of identified configuration parameters. Although notable differences were observed when comparing the classification performance across subsets grouped by annotation attribute values, they lack statistical significance under multiple testing correction. Nevertheless, achieving a balanced accuracy of 85.6%, this thesis demonstrates the promising potential of LLM-based citation verification while pointing out areas for future improvement.
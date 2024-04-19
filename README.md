# TitleTronLLM: From Abstracts to Elegant Titles

TitleTronLLM is a title generator for research papers designed to transform abstracts into concise and fitting titles. This tool utilizes Mistral-7B, an open-source Large Language Model (LLM), which has been fine-tuned on a dataset comprising research paper abstracts and their respective titles. The dataset was meticulously collected using the NASA-ADS API, ensuring quality and relevance. 

## How It Works

1. **Input**: Users provide an abstract of their research paper.
2. **Processing**: The abstract is fed into the Mistral-7B LLM.
3. **Output**: TitleTronLLM generates a suitable title based on the content of the abstract.

## Future Enhancements

Currently, the model is being further refined through fine-tuning on a dataset containing 200,000 samples. Additionally, Reinforcement Learning with Human Feedback (RLHF) is being employed for optimization purposes.

## Acknowledgments

Special thanks to NASA-ADS for providing the API used to collect the dataset for training the model.

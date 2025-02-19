## Collective Assignment
- Author: **Kirtan Mali**

### Introduction

Task: Remove any duplicate entries from the list provided in the [JSON](data/rawTrends.json) file.

Upon analysing the json file, I found that there are no exact duplicates in the file.
So, the task is not as simple as just removing duplicates.

For solving the problem I assume that removing duplicates means removing strings that mean the same things.  
___
### Setup using conda

```
conda create -n collective -c rapidsai -c conda-forge -c nvidia  \
    rapids=25.02 python=3.11 'cuda-version>=12.0,<=12.8'
pip3 install -r requirements.txt
```

For any other method refer to the [rapidsai](https://docs.rapids.ai/install/#conda) website
for installing cudf and cuml.
Then install the requirements using pip as shown above.
___
### Solution

- Approach 1  
  `Naively group strings based on the words occuring in them. Two strings having same word are grouped together.`  
  Results: [unique_entries_approach1.json](results/unique_entries_approach1.json)
  
- Approach 2  
  `Calculate embeddings for each string using TF-IDF. Apply PCA for dimensionality reduction. Then use these features to cluster close strings together`  
  Results: [unique_entries_approach2.json](results/unique_entries_approach2.json)

Code for both approaches along with its explanation is in the  [Jupyter Notebook](Remove%20Duplicates.ipynb).  
___
### Possible Improvements(which were not implemented)
  ```
  Enrich each entry in the list with LLM generated description with a prompt like
  """Describe the given words with less than 30 words.
     Reply with only the description.
     Reply should scritly be within 30 words.
     Reply must be as specific and concise as possible
     Topic: {topic}"""
  Once we have description for all trend topics, we can calculate the TF-IDF based on the desciptions
  and cluster based on these new embeddings.

  Using embeddings from a Embedding pretrained model.
  I tried using embeddings from a small model, but results weren't good.
  Maybe due to small size of model. I haven't investigated in depth on this.```

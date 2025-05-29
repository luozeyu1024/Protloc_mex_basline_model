# Protloc_mex_basline_model

Leveraging the advances introduced by the newest [ESMC](https://www.evolutionaryscale.ai/blog/esm-cambrian) model, we can  generalize our [feature-representation methods]( https://doi.org/10.1093/bib/bbad534)—including CLS/EOS token embeddings, global mean pooling, and residue-specific extraction—to this state-of-the-art model. 

Please note that the ESMC model is a protein-representation model developed by EvolutionaryScale (https://www.evolutionaryscale.ai/).
 We are merely extending the local sequence-window feature-extraction methods we previously built for ESM2 to this new architecture.
 For detailed information about the model, please consult the official documentation on evolutionaryscale.ai.

## Environment setting 

1. Install the environment for ESM C or ESM3

```
conda create -n esm_env_3C_fr python=3.10
```

```
conda activate esm_env_3C_fr 
```



### Download esm (Method 1)

```
pip install esm 
```

For more information please refer to the esm official [Github](https://github.com/evolutionaryscale/esm). if you have encounter install problem you can try below methods instead. 



### Install the esm source package (Method 2)

step1, download the ESMC github repository locally (https://github.com/evolutionaryscale/esm)

Then, find the `pyproject` file and modify its dependencies. Please adjust the dependencies according to your environment. Torch should be >=2.2.0, preferably the same version as installed in step 2. Then, remove the following dependencies:

```
Remove
"torch>=2.2.0", 
"torchvision",
"torchtext"
```

install pytorch manually

```
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

install torchtext manually

```
conda install -c pytorch torchtext
```

step2 install esm

First, cd to the path of the github package

```
cd ./esm-main/
```

Then install

```
pip install .
```



## ESMC Usage Example

Please refer to `ESMC_fr.py` line 389, the content after `if __name__ == '__main__':`. 

**Besides**, note the two parameters declared on **lines 392 and 393**—`WEIGHTS_PATH` and `TOKENIZER_PATH`.

- **`WEIGHTS_PATH`** designates the location of the **ESMC model checkpoint**.
  Download the complete model before use. The script currently supports the open-source [**ESMC-300 M**](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12) and [**ESMC-600 M**](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/tree/main) releases on Hugging Face.
- **`TOKENIZER_PATH`** points to an **ESM-2 configuration directory**.
  If you are running `ESMC_fr.py`, download the auxiliary files (e.g., `vocab.txt`) from *any* ESM-2 model—such as [**`esm2_t6_8M_UR50D`**](https://huggingface.co/facebook/esm2_t6_8M_UR50D)—and place them in this folder; having the model weights in the same directory does not cause problems.

This is different from the official model ESMC deploy.

## Citation

If our work has contributed to your research, we would greatly appreciate it if you could cite our work as follows.

Zeyu Luo, Rui Wang, Yawen Sun, Junhao Liu, Zongqing Chen, Yu-Juan Zhang, Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction, *Briefings in Bioinformatics*, Volume 25, Issue 2, March 2024, bbad534, https://doi.org/10.1093/bib/bbad534.

If you are using the ESM-2 OR ESMc model in your project or research, please refer to original work completed by the authors: 

**Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A.**
 *Evolutionary-scale prediction of atomic-level protein structure with a language model.*
 **Science**, 379(6637), 1123–1130 (2023).
 DOI: 10.1126/science.ade2574
 Preprint: bioRxiv 2022.07.20.500902

**Hayes, T., Rao, R., Akin, H., Sofroniew, N. J., Oktay, D., et al.**
 *Simulating 500 million years of evolution with a language model.*
 **Science**, 383(6671), 1234–1240 (2024).
 DOI: 10.1126/science.ads0018
 Preprint: bioRxiv 2024.07.02.123456

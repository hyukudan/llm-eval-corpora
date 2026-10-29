# Hugging Face dataset

Repository: dataset `hyukudan/llm-eval-corpora`, **our data only** (CC BY 4.0): synthetic and model-written
decision cases, writer prompt templates and the manifests. Third-party text (Gutenberg, Wikipedia, Aya,
oasst2, Dolly, MASSIVE, When2Call, source code) stays script-only: rebuild it with this repository.

Release steps:

1. `python verify.py --quick` (the decision files and their digests).
2. `python hf/assemble.py hf-out` (copies the files listed in `hf/assemble.py`, writes `SHA256SUMS`).
3. Upload: `hf upload hyukudan/llm-eval-corpora hf-out . --repo-type dataset`
   (the repository is created private first; visibility is changed by the owner).
4. Check that both configs load with `datasets.load_dataset("hyukudan/llm-eval-corpora", "<config>")`.

A held-back holdout set (ho4d) is added to the GitHub repository, not to this dataset, when its evaluations end.

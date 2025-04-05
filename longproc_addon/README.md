## LongProc Add-On on HELMET
We integrated [LongProc](https://github.com/princeton-pli/LongProc) in HELMET to support convenient evaluation.

**Additional Setup**
Pull the submodule from LongProc and add `__init__.py` files to make the import work:
```bash
git submodule update --init --recursive
touch longproc_addon/__init__.py
touch longproc_addon/longproc/__init__.py
```

To quickly test if everything is working, you can try running the evaluations.

**Running Evaluation**
You can now run evaluation just as you would in HELMET. The config files are stored in  `longproc_addon/configs`.

For example:
```bash
python eval.py --config longproc_addon/configs/html_to_tsv.yaml --model_name_or_path {local model path or huggingface model name} --output_dir {output directory, defaults to output/{model_name}}
```

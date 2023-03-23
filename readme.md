This is the repository for the "Clickbait Detection via Prompt-tuning with Titles Only".

First, by `pip install -r requirement.txt` to install all the dependencies.

Then you need to install OpenPrompt.
And copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py.

In addition,You need to use e methods to filter tag words,In this process, you need to download the corresponding vocabulary.
These words need to be put scripts/TextClassification，and modify the corresponding position.

Also, you can put your own dataset in datasets/TextClassification.

example shell scripts:

python main.py  --verbalizer cpt




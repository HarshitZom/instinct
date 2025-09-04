# Instinct, Continue's Open Next Edit Model

This repo contains code used to train and evaluate Continue's [Instinct Next Edit model](https://huggingface.co/continuedev/instinct). Learn more about Instinct [here](https://blog.continue.dev/instinct/).

To install dependencies, run:

```
pip install uv
uv pip install -r requirements.txt
uv pip install --no-build-isolation flash-attn==2.5.8
```

The fine-tuning script is located in the `sft` directory and run via `launch_sft.sh`. Please feel free to work off of this code to enhance Instinct! We welcome contributions.
